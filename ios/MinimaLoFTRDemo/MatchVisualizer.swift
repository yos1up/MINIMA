// Draws the two source images side-by-side and overlays match lines colored by
// confidence. Returns a UIImage suitable for SwiftUI `Image(uiImage:)`.

import UIKit

enum MatchVisualizer {
    static func render(image0: UIImage,
                       image1: UIImage,
                       matches: [LoFTRMatch],
                       maxLines: Int = 200) -> UIImage? {
        let h = max(image0.size.height, image1.size.height)
        let w = image0.size.width + image1.size.width
        let canvasSize = CGSize(width: w, height: h)

        let renderer = UIGraphicsImageRenderer(size: canvasSize)
        return renderer.image { ctx in
            UIColor.black.setFill()
            ctx.fill(CGRect(origin: .zero, size: canvasSize))
            image0.draw(in: CGRect(x: 0, y: 0, width: image0.size.width, height: image0.size.height))
            image1.draw(in: CGRect(x: image0.size.width, y: 0,
                                   width: image1.size.width, height: image1.size.height))

            let drawn = Array(matches.prefix(maxLines))
            for m in drawn {
                let p0 = m.p0
                let p1 = CGPoint(x: m.p1.x + image0.size.width, y: m.p1.y)
                let color = colorForConfidence(m.confidence)
                color.setStroke()
                color.setFill()
                let path = UIBezierPath()
                path.move(to: p0)
                path.addLine(to: p1)
                path.lineWidth = 1.0
                path.stroke()
                let r: CGFloat = 2
                ctx.cgContext.fillEllipse(in: CGRect(x: p0.x - r, y: p0.y - r, width: 2 * r, height: 2 * r))
                ctx.cgContext.fillEllipse(in: CGRect(x: p1.x - r, y: p1.y - r, width: 2 * r, height: 2 * r))
            }
        }
    }

    /// Simple jet-style colormap: low conf = blue, mid = green, high = red.
    private static func colorForConfidence(_ c: Float) -> UIColor {
        // Map [thr=0.2, 1.0] -> [0, 1].
        let t = max(0, min(1, (CGFloat(c) - 0.2) / 0.8))
        let r = max(0, min(1, 1.5 - abs(4.0 * t - 3.0)))
        let g = max(0, min(1, 1.5 - abs(4.0 * t - 2.0)))
        let b = max(0, min(1, 1.5 - abs(4.0 * t - 1.0)))
        return UIColor(red: r, green: g, blue: b, alpha: 1.0)
    }
}
