// CardBackgroundView.swift renders the visual background for cards,
// including images, lined paper, and grid patterns with margin support.
import SwiftUI

struct CardBackgroundView: View {
    let background: CardBackground
    let size: CGSize

    // Calculate actual margin sizes based on percentages
    private var marginSizes: (left: CGFloat, right: CGFloat, top: CGFloat, bottom: CGFloat) {
        (
            left: background.margins.left.isEnabled ? size.width * (background.margins.left.percentage / 100) : 0,
            right: background.margins.right.isEnabled ? size.width * (background.margins.right.percentage / 100) : 0,
            top: background.margins.top.isEnabled ? size.height * (background.margins.top.percentage / 100) : 0,
            bottom: background.margins.bottom.isEnabled ? size.height * (background.margins.bottom.percentage / 100) : 0
        )
    }

    var body: some View {
        Canvas { context, _ in
            switch background.style {
            case .none:
                break

            case .solidColor(let codableColor):
                // Fill with solid color
                let rect = CGRect(origin: .zero, size: size)
                context.fill(Path(rect), with: .color(codableColor.color))

            case .grid:
                drawGrid(context: context, size: size)

            case .lined:
                drawLines(context: context, size: size)

            case .image:
                if let imageData = background.imageData,
                   let uiImage = UIImage(data: imageData) {
                    let image = Image(uiImage: uiImage)
                    context.opacity = background.imageOpacity
                    context.draw(image, in: CGRect(origin: .zero, size: size))
                }
            }
        }
        .frame(width: size.width, height: size.height)
    }

    private func drawGrid(context: GraphicsContext, size: CGSize) {
        let margins = marginSizes
        let path = Path { path in
            // Draw margins first
            drawMargins(path: &path, size: size)

            // Calculate grid boundaries respecting margins
            let startX = margins.left
            let endX = size.width - margins.right
            let startY = margins.top
            let endY = size.height - margins.bottom

            // Vertical lines
            var x = startX
            while x <= endX {
                path.move(to: CGPoint(x: x, y: startY))
                path.addLine(to: CGPoint(x: x, y: endY))
                x += background.spacing
            }

            // Horizontal lines
            var y = startY
            while y <= endY {
                path.move(to: CGPoint(x: startX, y: y))
                path.addLine(to: CGPoint(x: endX, y: y))
                y += background.spacing
            }
        }

        context.stroke(path, with: .color(background.lineColor.color), lineWidth: background.lineWidth)
    }

    private func drawLines(context: GraphicsContext, size: CGSize) {
        let margins = marginSizes
        let path = Path { path in
            // Draw margins first
            drawMargins(path: &path, size: size)

            // Calculate line boundaries
            let startX = margins.left
            let endX = size.width - margins.right
            let startY = margins.top
            let endY = size.height - margins.bottom

            // Draw horizontal lines
            var y = startY
            while y <= endY {
                // Skip if we're exactly on a margin
                let isOnTopMargin = background.margins.top.isEnabled && abs(y - margins.top) < 0.1
                let isOnBottomMargin = background.margins.bottom.isEnabled && abs(y - (size.height - margins.bottom)) < 0.1

                if !isOnTopMargin && !isOnBottomMargin {
                    path.move(to: CGPoint(x: startX, y: y))
                    path.addLine(to: CGPoint(x: endX, y: y))
                }

                y += background.spacing
            }
        }

        context.stroke(path, with: .color(background.lineColor.color), lineWidth: background.lineWidth)
    }

    private func drawMargins(path: inout Path, size: CGSize) {
        let margins = marginSizes

        // Left margin
        if background.margins.left.isEnabled {
            path.move(to: CGPoint(x: margins.left, y: 0))
            path.addLine(to: CGPoint(x: margins.left, y: size.height))
        }

        // Right margin
        if background.margins.right.isEnabled {
            let x = size.width - margins.right
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: size.height))
        }

        // Top margin
        if background.margins.top.isEnabled {
            path.move(to: CGPoint(x: 0, y: margins.top))
            path.addLine(to: CGPoint(x: size.width, y: margins.top))
        }

        // Bottom margin
        if background.margins.bottom.isEnabled {
            let y = size.height - margins.bottom
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: size.width, y: y))
        }
    }
}
