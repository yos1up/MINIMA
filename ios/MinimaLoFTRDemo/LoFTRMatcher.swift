// Orchestrates the two ONNX sessions and the CPU-side glue.
//
// Usage:
//     let matcher = try LoFTRMatcher()
//     let result = try matcher.match(image0: uiImage0, image1: uiImage1)
//     print("\(result.matches.count) matches in \(result.totalMs) ms")

import Foundation
import UIKit
import onnxruntime_objc

struct LoFTRMatch {
    let p0: CGPoint  // pixel coords in original image0
    let p1: CGPoint  // pixel coords in original image1
    let confidence: Float
}

struct LoFTRMatchResult {
    let matches: [LoFTRMatch]
    let preprocessMs: Double
    let stage1Ms: Double
    let coarseMs: Double
    let stage2Ms: Double
    var totalMs: Double { preprocessMs + stage1Ms + coarseMs + stage2Ms }
}

final class LoFTRMatcher {
    // Fixed at export time — must match `--img-size` used in mobile/export_onnx.py.
    let imgSize: Int = 640
    var Hc: Int { imgSize / 8 }
    var Wc: Int { imgSize / 8 }
    var Hf: Int { imgSize / 2 }
    var Wf: Int { imgSize / 2 }
    let coarseChannels = 256
    let fineChannels = 128

    private let env: ORTEnv
    private let stage1: ORTSession
    private let stage2: ORTSession

    init(stage1Resource: String = "loftr_stage1_640",
         stage2Resource: String = "loftr_stage2") throws {
        env = try ORTEnv(loggingLevel: .warning)

        let opt = try ORTSessionOptions()
        try opt.setIntraOpNumThreads(4)
        try opt.setGraphOptimizationLevel(.all)

        guard let s1Path = Bundle.main.path(forResource: stage1Resource, ofType: "onnx") else {
            throw MatcherError.modelMissing(stage1Resource)
        }
        guard let s2Path = Bundle.main.path(forResource: stage2Resource, ofType: "onnx") else {
            throw MatcherError.modelMissing(stage2Resource)
        }
        stage1 = try ORTSession(env: env, modelPath: s1Path, sessionOptions: opt)
        stage2 = try ORTSession(env: env, modelPath: s2Path, sessionOptions: opt)
    }

    enum MatcherError: Error, CustomStringConvertible {
        case modelMissing(String)
        case preprocessFailed
        case unexpectedShape(String)
        var description: String {
            switch self {
            case .modelMissing(let n): return "ONNX model \(n).onnx not found in app bundle. Did you drag it into Xcode with 'Copy if needed'?"
            case .preprocessFailed: return "Image preprocessing failed."
            case .unexpectedShape(let s): return "Unexpected tensor shape: \(s)"
            }
        }
    }

    func match(image0: UIImage, image1: UIImage,
               threshold: Float = 0.2,
               borderRm: Int = 2,
               temperature: Float = 0.1) throws -> LoFTRMatchResult {
        let t0 = Date()
        guard let pre0 = ImagePreprocess.process(image0, targetSize: imgSize),
              let pre1 = ImagePreprocess.process(image1, targetSize: imgSize),
              pre0.height == imgSize, pre0.width == imgSize,
              pre1.height == imgSize, pre1.width == imgSize
        else { throw MatcherError.preprocessFailed }
        let preprocessMs = Date().timeIntervalSince(t0) * 1000

        // ---- Stage 1: image pair -> coarse and fine features ----
        let t1 = Date()
        let s1Outputs = try runStage1(pre0: pre0, pre1: pre1)
        let stage1Ms = Date().timeIntervalSince(t1) * 1000

        // ---- CPU: dual-softmax + mutual NN + border ----
        let t2 = Date()
        let coarseMatches = CoarseMatching.selectMatches(
            featC0: s1Outputs.featC0,
            featC1: s1Outputs.featC1,
            gridHeight: Hc, gridWidth: Wc, channels: coarseChannels,
            threshold: threshold, borderRm: borderRm, temperature: temperature
        )
        let coarseMs = Date().timeIntervalSince(t2) * 1000

        if coarseMatches.isEmpty {
            return LoFTRMatchResult(matches: [], preprocessMs: preprocessMs,
                                    stage1Ms: stage1Ms, coarseMs: coarseMs, stage2Ms: 0)
        }

        // ---- Stage 2: fine windows + coarse context -> sub-pixel offset ----
        let t3 = Date()
        let iIds = coarseMatches.map(\.iId)
        let jIds = coarseMatches.map(\.jId)
        let win0 = FineWindowCrop.crop(
            featF: s1Outputs.featF0, channels: fineChannels,
            fineHeight: Hf, fineWidth: Wf, gridHeight: Hc, gridWidth: Wc,
            flatIds: iIds, useImage0Side: true)
        let win1 = FineWindowCrop.crop(
            featF: s1Outputs.featF1, channels: fineChannels,
            fineHeight: Hf, fineWidth: Wf, gridHeight: Hc, gridWidth: Wc,
            flatIds: jIds, useImage0Side: false)
        let pickC0 = FineWindowCrop.gatherCoarse(featC: s1Outputs.featC0, channels: coarseChannels, flatIds: iIds)
        let pickC1 = FineWindowCrop.gatherCoarse(featC: s1Outputs.featC1, channels: coarseChannels, flatIds: jIds)

        let expec = try runStage2(win0: win0, win1: win1, pickC0: pickC0, pickC1: pickC1,
                                  M: coarseMatches.count)
        let stage2Ms = Date().timeIntervalSince(t3) * 1000

        // ---- Assemble final mkpts in original image pixel coords ----
        let coarseScale: Float = Float(imgSize) / Float(Hc)  // 8
        let fineHalf: Float = Float(FineWindowCrop.windowSize / 2)  // 2
        let fineScale: Float = Float(imgSize) / Float(Hf)  // 2
        let sx0 = Float(pre0.scale.x), sy0 = Float(pre0.scale.y)
        let sx1 = Float(pre1.scale.x), sy1 = Float(pre1.scale.y)

        var matches: [LoFTRMatch] = []
        matches.reserveCapacity(coarseMatches.count)
        for (m, cm) in coarseMatches.enumerated() {
            let xi = Float(cm.iId % Wc) * coarseScale
            let yi = Float(cm.iId / Wc) * coarseScale
            let xj = Float(cm.jId % Wc) * coarseScale + expec[m * 2 + 0] * fineHalf * fineScale
            let yj = Float(cm.jId / Wc) * coarseScale + expec[m * 2 + 1] * fineHalf * fineScale
            matches.append(LoFTRMatch(
                p0: CGPoint(x: CGFloat(xi * sx0), y: CGFloat(yi * sy0)),
                p1: CGPoint(x: CGFloat(xj * sx1), y: CGFloat(yj * sy1)),
                confidence: cm.conf
            ))
        }

        return LoFTRMatchResult(matches: matches,
                                preprocessMs: preprocessMs,
                                stage1Ms: stage1Ms,
                                coarseMs: coarseMs,
                                stage2Ms: stage2Ms)
    }

    // ---- ORT plumbing ----

    private struct Stage1Outputs {
        let featC0: [Float]  // (L, C_c)
        let featC1: [Float]
        let featF0: [Float]  // (C_f, Hf, Wf)
        let featF1: [Float]
    }

    private func runStage1(pre0: PreprocessResult, pre1: PreprocessResult) throws -> Stage1Outputs {
        let shape: [NSNumber] = [1, 1, NSNumber(value: imgSize), NSNumber(value: imgSize)]
        let val0 = try makeTensor(from: pre0.pixels, shape: shape)
        let val1 = try makeTensor(from: pre1.pixels, shape: shape)
        let names: Set<String> = ["feat_c0", "feat_c1", "feat_f0", "feat_f1"]
        let outputs = try stage1.run(
            withInputs: ["image0": val0, "image1": val1],
            outputNames: names,
            runOptions: nil
        )
        return Stage1Outputs(
            featC0: try floatArray(from: outputs["feat_c0"]!),
            featC1: try floatArray(from: outputs["feat_c1"]!),
            featF0: try floatArray(from: outputs["feat_f0"]!),
            featF1: try floatArray(from: outputs["feat_f1"]!)
        )
    }

    private func runStage2(win0: [Float], win1: [Float],
                           pickC0: [Float], pickC1: [Float],
                           M: Int) throws -> [Float] {
        let WW = FineWindowCrop.windowSize * FineWindowCrop.windowSize
        let winShape: [NSNumber] = [NSNumber(value: M), NSNumber(value: WW), NSNumber(value: fineChannels)]
        let pickShape: [NSNumber] = [NSNumber(value: M), NSNumber(value: coarseChannels)]
        let v0 = try makeTensor(from: win0, shape: winShape)
        let v1 = try makeTensor(from: win1, shape: winShape)
        let p0 = try makeTensor(from: pickC0, shape: pickShape)
        let p1 = try makeTensor(from: pickC1, shape: pickShape)
        let out = try stage2.run(
            withInputs: ["feat_f0_win": v0, "feat_f1_win": v1,
                         "feat_c0_pick": p0, "feat_c1_pick": p1],
            outputNames: ["expec_f"],
            runOptions: nil
        )
        return try floatArray(from: out["expec_f"]!)
    }

    private func makeTensor(from floats: [Float], shape: [NSNumber]) throws -> ORTValue {
        let byteCount = floats.count * MemoryLayout<Float>.size
        let data = NSMutableData(length: byteCount)!
        floats.withUnsafeBufferPointer { src in
            memcpy(data.mutableBytes, src.baseAddress!, byteCount)
        }
        return try ORTValue(tensorData: data, elementType: .float, shape: shape)
    }

    private func floatArray(from value: ORTValue) throws -> [Float] {
        let info = try value.tensorTypeAndShapeInfo()
        let count: Int = info.shape.reduce(1) { $0 * $1.intValue }
        let data = try value.tensorData() as Data
        guard data.count == count * MemoryLayout<Float>.size else {
            throw MatcherError.unexpectedShape("byte mismatch")
        }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
