// CPU-side coarse match selection — Swift port of
// mobile/pipeline.py::select_coarse_matches.
//
// Given feat_c0, feat_c1 (post-coarse-transformer, shape [L, C]):
//   1. normalize by sqrt(C)
//   2. sim = (f0 @ f1^T) / temperature                 -- (L, L) via cblas_sgemm
//   3. conf = softmax(sim, dim=0) * softmax(sim, dim=1)
//   4. zero-out cells within `borderRm` of the (Hc, Wc) grid border
//   5. keep (i, j) where conf[i, j] is the row max AND the col max AND > thr
//
// For default 640x640 input, L = 80*80 = 6400, so the transient conf matrix
// is 6400 * 6400 * 4B ≈ 164 MB. Fine on any modern iPhone.

import Accelerate
import Foundation

struct CoarseMatch {
    let iId: Int  // flattened (Hc, Wc) index for image0
    let jId: Int  // flattened (Hc, Wc) index for image1
    let conf: Float
}

enum CoarseMatching {
    static func selectMatches(
        featC0: [Float],
        featC1: [Float],
        gridHeight Hc: Int,
        gridWidth Wc: Int,
        channels C: Int,
        threshold thr: Float = 0.2,
        borderRm: Int = 2,
        temperature: Float = 0.1
    ) -> [CoarseMatch] {
        let L = Hc * Wc
        precondition(featC0.count == L * C)
        precondition(featC1.count == L * C)

        // 1. Normalize. Fold 1/sqrt(C) into the matmul scalar (along with 1/temperature).
        let invSqrtC = 1.0 / sqrtf(Float(C))
        let alpha = (invSqrtC * invSqrtC) / temperature  // = 1 / (C * temperature)

        // 2. sim = alpha * (f0 @ f1^T). cblas_sgemm: row-major, A is L x C, B is L x C (transposed),
        //    so M=L, N=L, K=C, lda=C, ldb=C, ldc=L.
        var sim = [Float](repeating: 0, count: L * L)
        featC0.withUnsafeBufferPointer { a in
            featC1.withUnsafeBufferPointer { b in
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(L), Int32(L), Int32(C),
                    alpha,
                    a.baseAddress, Int32(C),
                    b.baseAddress, Int32(C),
                    0.0,
                    &sim, Int32(L)
                )
            }
        }

        // 3a. softmax along dim=1 (per row): subtract rowmax, exp, normalize by rowsum.
        var softRow = [Float](repeating: 0, count: L * L)
        rowwiseSoftmax(into: &softRow, from: sim, rows: L, cols: L)

        // 3b. softmax along dim=0 (per col): subtract colmax, exp, normalize by colsum.
        var softCol = [Float](repeating: 0, count: L * L)
        columnwiseSoftmax(into: &softCol, from: sim, rows: L, cols: L)

        // 3c. conf = softRow * softCol elementwise (overwrites softRow).
        vDSP_vmul(softRow, 1, softCol, 1, &softRow, 1, vDSP_Length(L * L))
        var conf = softRow  // alias for clarity
        softCol = []
        sim = []

        // 4. border mask in coarse-grid space.
        if borderRm > 0 {
            applyBorderMask(&conf, Hc: Hc, Wc: Wc, border: borderRm)
        }

        // 5. mutual NN + threshold.
        return mutualNN(conf: conf, L: L, threshold: thr)
    }

    // ---- helpers ----

    private static func rowwiseSoftmax(into out: inout [Float], from input: [Float], rows R: Int, cols C: Int) {
        // For each row independently:
        //   m = max(row); row -= m; exp(row); row /= sum(row)
        var n32 = Int32(C)
        for i in 0..<R {
            let off = i * C
            var m: Float = 0
            input.withUnsafeBufferPointer { p in
                vDSP_maxv(p.baseAddress!.advanced(by: off), 1, &m, vDSP_Length(C))
            }
            var negM = -m
            input.withUnsafeBufferPointer { p in
                vDSP_vsadd(p.baseAddress!.advanced(by: off), 1, &negM, &out[off], 1, vDSP_Length(C))
            }
            vvexpf(&out[off], &out[off], &n32)
            var s: Float = 0
            vDSP_sve(&out[off], 1, &s, vDSP_Length(C))
            var inv = 1.0 / s
            vDSP_vsmul(&out[off], 1, &inv, &out[off], 1, vDSP_Length(C))
        }
    }

    private static func columnwiseSoftmax(into out: inout [Float], from input: [Float], rows R: Int, cols C: Int) {
        // Strided ops along each column (stride = C, length = R).
        var n32 = Int32(R)
        for j in 0..<C {
            var m: Float = 0
            input.withUnsafeBufferPointer { p in
                vDSP_maxv(p.baseAddress!.advanced(by: j), C, &m, vDSP_Length(R))
            }
            // out[i, j] = input[i, j] - m  for i in 0..<R
            var negM = -m
            input.withUnsafeBufferPointer { src in
                out.withUnsafeMutableBufferPointer { dst in
                    vDSP_vsadd(src.baseAddress!.advanced(by: j), C, &negM,
                               dst.baseAddress!.advanced(by: j), C, vDSP_Length(R))
                }
            }
            // exp in place (strided exp is not supported by vvexpf; gather, exp, scatter).
            var col = [Float](repeating: 0, count: R)
            for i in 0..<R { col[i] = out[i * C + j] }
            vvexpf(&col, col, &n32)
            var s: Float = 0
            vDSP_sve(col, 1, &s, vDSP_Length(R))
            let inv: Float = 1.0 / s
            for i in 0..<R { out[i * C + j] = col[i] * inv }
        }
    }

    private static func applyBorderMask(_ conf: inout [Float], Hc: Int, Wc: Int, border b: Int) {
        let L = Hc * Wc
        // Mark border cells in the (Hc, Wc) coarse grid.
        var isBorder = [Bool](repeating: false, count: L)
        for h in 0..<Hc {
            for w in 0..<Wc {
                if h < b || h >= Hc - b || w < b || w >= Wc - b {
                    isBorder[h * Wc + w] = true
                }
            }
        }
        // Zero rows and columns whose id is a border cell.
        for id in 0..<L where isBorder[id] {
            for j in 0..<L { conf[id * L + j] = 0 }
        }
        for id in 0..<L where isBorder[id] {
            for i in 0..<L { conf[i * L + id] = 0 }
        }
    }

    private static func mutualNN(conf: [Float], L: Int, threshold thr: Float) -> [CoarseMatch] {
        // We need row argmax and column argmax. vDSP_maxvi's index semantics
        // under stride > 1 are inconsistent across documentation revisions, so
        // we sweep with plain loops. Both passes together cost ~80 ms at L=6400
        // on an A15.
        var rowArgmax = [Int](repeating: 0, count: L)
        var colArgmax = [Int](repeating: 0, count: L)
        var colBest = [Float](repeating: -.greatestFiniteMagnitude, count: L)

        conf.withUnsafeBufferPointer { p in
            let base = p.baseAddress!
            for i in 0..<L {
                let row = base.advanced(by: i * L)
                var bestVal: Float = -.greatestFiniteMagnitude
                var bestIdx = 0
                for j in 0..<L {
                    let v = row[j]
                    if v > bestVal { bestVal = v; bestIdx = j }
                    if v > colBest[j] { colBest[j] = v; colArgmax[j] = i }
                }
                rowArgmax[i] = bestIdx
            }
        }

        var matches: [CoarseMatch] = []
        matches.reserveCapacity(2048)
        for i in 0..<L {
            let j = rowArgmax[i]
            guard colArgmax[j] == i else { continue }
            let c = conf[i * L + j]
            guard c > thr else { continue }
            matches.append(CoarseMatch(iId: i, jId: j, conf: c))
        }
        return matches
    }
}
