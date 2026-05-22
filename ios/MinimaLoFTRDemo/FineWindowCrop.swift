// Per-match fine-level window extraction — Swift port of
// mobile/pipeline.py::crop_fine_windows.
//
// For each coarse match (Hc-cell, Wc-cell), copy a 5x5 window of the (C, Hf, Wf)
// fine feature map centered at (cell.h * stride, cell.w * stride) with zero
// padding of W//2 = 2.  Output layout: (M, W*W, C) — channel-last, the shape
// stage 2's ONNX expects.

import Foundation

enum FineWindowCrop {
    static let windowSize = 5

    /// Returns a flat Float buffer of size `matches.count * W*W * channels`,
    /// laid out (M, W*W, C). Use this directly as stage-2 input.
    static func crop(
        featF: [Float],   // (C, Hf, Wf), row-major
        channels C: Int,
        fineHeight Hf: Int,
        fineWidth Wf: Int,
        gridHeight Hc: Int,
        gridWidth Wc: Int,
        flatIds: [Int],
        useImage0Side: Bool  // unused; kept for API symmetry with two-side calls
    ) -> [Float] {
        _ = useImage0Side
        let W = windowSize
        let pad = W / 2
        let stride = Hf / Hc
        let M = flatIds.count
        var out = [Float](repeating: 0, count: M * W * W * C)
        if M == 0 { return out }

        // We zero-pad on the fly during the read by clamping indices and
        // returning 0 for out-of-bounds positions. Avoids allocating a
        // (C, HfP, WfP) padded copy (8x the channel count makes this big).
        featF.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for (mi, id) in flatIds.enumerated() {
                    let hc = id / Wc
                    let wc = id % Wc
                    let hTopPadded = hc * stride                 // inclusive
                    let wLeftPadded = wc * stride                // inclusive
                    // padded coords -> real coords: subtract `pad`.
                    for dh in 0..<W {
                        let hReal = hTopPadded + dh - pad
                        for dw in 0..<W {
                            let wReal = wLeftPadded + dw - pad
                            let outBase = ((mi * W * W) + (dh * W) + dw) * C
                            if hReal < 0 || hReal >= Hf || wReal < 0 || wReal >= Wf {
                                // Already zero (we initialized to zeros).
                                continue
                            }
                            // featF[c, hReal, wReal] across all C channels:
                            // featF index = c * (Hf * Wf) + hReal * Wf + wReal
                            let plane = Hf * Wf
                            for c in 0..<C {
                                dst[outBase + c] = src[c * plane + hReal * Wf + wReal]
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// Gather coarse-feature rows at the given flat indices.
    /// `featC` is (L, C) row-major; result is (M, C).
    static func gatherCoarse(featC: [Float], channels C: Int, flatIds: [Int]) -> [Float] {
        var out = [Float](repeating: 0, count: flatIds.count * C)
        featC.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for (mi, id) in flatIds.enumerated() {
                    let from = id * C
                    let to = mi * C
                    for c in 0..<C { dst[to + c] = src[from + c] }
                }
            }
        }
        return out
    }
}
