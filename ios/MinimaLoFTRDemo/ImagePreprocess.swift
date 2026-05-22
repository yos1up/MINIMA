// Turn a UIImage into the (1, 1, H, W) float32 grayscale tensor that
// loftr_stage1 expects. Matches the numpy reference in mobile/pipeline.py.
//
// - Convert to grayscale (BT.601 luma).
// - Resize so the longer side equals `targetSize`, snap each side down to a
//   multiple of `df = 8`.
// - Optionally zero-pad to a `targetSize x targetSize` square (the default).
// - Normalize to [0, 1] float32, stored row-major in a contiguous Float array.

import Accelerate
import CoreGraphics
import UIKit

struct PreprocessResult {
    /// Row-major float32 buffer of shape (H, W) with values in [0, 1].
    let pixels: [Float]
    /// Tensor side (H == W when padded).
    let height: Int
    let width: Int
    /// (sx, sy) = original-pixel-per-network-pixel. Multiply network-space
    /// keypoints by this to recover original-image coordinates.
    let scale: CGPoint
}

enum ImagePreprocess {
    static func process(
        _ image: UIImage,
        targetSize: Int = 640,
        divisibleFactor df: Int = 8,
        padToSquare: Bool = true
    ) -> PreprocessResult? {
        guard let cgImage = image.cgImage else { return nil }
        let originalW = cgImage.width
        let originalH = cgImage.height

        // Compute resized dimensions (snap to multiple of df).
        let s = Double(targetSize) / Double(max(originalW, originalH))
        var newW = max(df, (Int(Double(originalW) * s) / df) * df)
        var newH = max(df, (Int(Double(originalH) * s) / df) * df)
        if padToSquare {
            // We render into a targetSize x targetSize canvas with origin at top-left.
            // Image goes into the (newH x newW) region; the rest stays zero.
            // The longer side equals targetSize by construction.
        }

        let canvasW = padToSquare ? targetSize : newW
        let canvasH = padToSquare ? targetSize : newH

        // Render into a single-channel (grayscale) 8-bit context. CoreGraphics
        // will downsample and convert color in one shot.
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(
            data: nil,
            width: canvasW,
            height: canvasH,
            bitsPerComponent: 8,
            bytesPerRow: canvasW,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }

        // CoreGraphics is bottom-left origin. Flip the Y axis so the rect
        // below uses top-left (image) coordinates and lands the resized
        // bitmap at the top-left of the canvas.
        ctx.translateBy(x: 0, y: CGFloat(canvasH))
        ctx.scaleBy(x: 1, y: -1)
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: CGFloat(newW), height: CGFloat(newH)))

        guard let buffer = ctx.data else { return nil }
        let bytes = buffer.assumingMemoryBound(to: UInt8.self)

        // Convert UInt8 [0, 255] → Float32 [0, 1] with vDSP.
        var pixels = [Float](repeating: 0, count: canvasW * canvasH)
        vDSP.convertElements(of: UnsafeBufferPointer(start: bytes, count: canvasW * canvasH), to: &pixels)
        var scale: Float = 1.0 / 255.0
        vDSP_vsmul(pixels, 1, &scale, &pixels, 1, vDSP_Length(pixels.count))

        // CoreGraphics drew the image at the TOP of the canvas above, but we
        // want it at the top-left of a Y-down tensor coordinate system. The
        // flip we already did handles that. Sanity: pixels[0] is top-left.

        return PreprocessResult(
            pixels: pixels,
            height: canvasH,
            width: canvasW,
            scale: CGPoint(x: Double(originalW) / Double(newW), y: Double(originalH) / Double(newH))
        )
    }
}
