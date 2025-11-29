// CardBackground.swift defines the data model for card backgrounds,
// including support for solid colors, images, lined paper, and grid patterns.
import Foundation
import SwiftUI
import simd

// MARK: - Background Style

enum CardBackgroundStyle: Codable, Equatable {
    case none
    case solidColor(color: CodableColor)
    case image
    case lined
    case grid

    // Codable conformance for enum with associated values
    enum CodingKeys: String, CodingKey {
        case type
        case color
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "none":
            self = .none
        case "solidColor":
            let color = try container.decode(CodableColor.self, forKey: .color)
            self = .solidColor(color: color)
        case "image":
            self = .image
        case "lined":
            self = .lined
        case "grid":
            self = .grid
        default:
            self = .none
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .none:
            try container.encode("none", forKey: .type)
        case .solidColor(let color):
            try container.encode("solidColor", forKey: .type)
            try container.encode(color, forKey: .color)
        case .image:
            try container.encode("image", forKey: .type)
        case .lined:
            try container.encode("lined", forKey: .type)
        case .grid:
            try container.encode("grid", forKey: .type)
        }
    }
}

// MARK: - Margin Configuration

struct MarginConfig: Codable, Equatable {
    var isEnabled: Bool
    var percentage: CGFloat  // 0-100, represents percentage of card dimension

    init(isEnabled: Bool = false, percentage: CGFloat = 10.0) {
        self.isEnabled = isEnabled
        self.percentage = percentage
    }
}

struct MarginsConfig: Codable, Equatable {
    var left: MarginConfig
    var right: MarginConfig
    var top: MarginConfig
    var bottom: MarginConfig

    init(left: MarginConfig = MarginConfig(),
         right: MarginConfig = MarginConfig(),
         top: MarginConfig = MarginConfig(),
         bottom: MarginConfig = MarginConfig()) {
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
    }

    // Preset: Standard lined paper with left margin
    static var linedPaper: MarginsConfig {
        MarginsConfig(
            left: MarginConfig(isEnabled: true, percentage: 15.0),
            right: MarginConfig(isEnabled: false),
            top: MarginConfig(isEnabled: false),
            bottom: MarginConfig(isEnabled: false)
        )
    }
}

// MARK: - Codable Color

struct CodableColor: Codable, Equatable {
    var red: Double
    var green: Double
    var blue: Double
    var alpha: Double

    init(red: Double, green: Double, blue: Double, alpha: Double = 1.0) {
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
    }

    init(_ color: Color) {
        // Extract RGBA components from Color (approximation for common cases)
        // For production, you might want to use UIColor/NSColor
        self.red = 0.0
        self.green = 0.0
        self.blue = 0.0
        self.alpha = 1.0
    }

    var color: Color {
        Color(red: red, green: green, blue: blue, opacity: alpha)
    }

    static var blue: CodableColor {
        CodableColor(red: 0.0, green: 0.47, blue: 1.0, alpha: 1.0)
    }

    static var lightGray: CodableColor {
        CodableColor(red: 0.8, green: 0.8, blue: 0.8, alpha: 1.0)
    }

    static var black: CodableColor {
        CodableColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
    }
}

// MARK: - CardBackground

class CardBackground: Codable, Equatable {
    var style: CardBackgroundStyle

    // Image properties
    var imageData: Data?
    var imageOpacity: Double

    // Line/Grid properties
    var lineColor: CodableColor
    var lineWidth: CGFloat
    var spacing: CGFloat  // Distance between lines/grid cells
    var margins: MarginsConfig

    init(style: CardBackgroundStyle = .none) {
        self.style = style
        self.imageData = nil
        self.imageOpacity = 1.0
        self.lineColor = .lightGray
        self.lineWidth = 1.0
        self.spacing = 30.0  // Default spacing (30pt between lines)
        self.margins = MarginsConfig()
    }

    // Preset: Standard lined paper
    static func linedPaper() -> CardBackground {
        let bg = CardBackground(style: .lined)
        bg.lineColor = .blue
        bg.lineWidth = 1.0
        bg.spacing = 25.0  // Typical lined paper spacing
        bg.margins = .linedPaper
        return bg
    }

    // Preset: Grid paper
    static func gridPaper() -> CardBackground {
        let bg = CardBackground(style: .grid)
        bg.lineColor = .lightGray
        bg.lineWidth = 0.5
        bg.spacing = 20.0  // Grid cell size
        bg.margins = MarginsConfig()
        return bg
    }

    // Equatable conformance
    static func == (lhs: CardBackground, rhs: CardBackground) -> Bool {
        lhs.style == rhs.style &&
        lhs.imageData == rhs.imageData &&
        lhs.imageOpacity == rhs.imageOpacity &&
        lhs.lineColor == rhs.lineColor &&
        lhs.lineWidth == rhs.lineWidth &&
        lhs.spacing == rhs.spacing &&
        lhs.margins == rhs.margins
    }

    /// Generate Metal uniforms for procedural shader rendering
    /// - Parameter cardSize: The size of the card in world units
    /// - Returns: Uniforms struct matching the Metal shader definition
    func makeUniforms(cardSize: SIMD2<Double>) -> CardBackgroundUniforms {
        // Get base color (default to white if solid color not set)
        var baseColor = SIMD4<Float>(1.0, 1.0, 1.0, 1.0)
        if case .solidColor(let color) = style {
            baseColor = SIMD4<Float>(
                Float(color.red),
                Float(color.green),
                Float(color.blue),
                Float(color.alpha)
            )
        }

        // Convert line color
        let lineColorSIMD = SIMD4<Float>(
            Float(lineColor.red),
            Float(lineColor.green),
            Float(lineColor.blue),
            Float(lineColor.alpha)
        )

        // Calculate margin percentages (0.0 - 1.0)
        let marginTopPct = margins.top.isEnabled ? Float(margins.top.percentage / 100.0) : 0.0
        let marginBottomPct = margins.bottom.isEnabled ? Float(margins.bottom.percentage / 100.0) : 0.0
        let marginLeftPct = margins.left.isEnabled ? Float(margins.left.percentage / 100.0) : 0.0
        let marginRightPct = margins.right.isEnabled ? Float(margins.right.percentage / 100.0) : 0.0

        // Determine style integer
        let styleInt: Int32
        switch style {
        case .none, .solidColor:
            styleInt = 0  // Solid
        case .lined:
            styleInt = 1  // Lined
        case .grid:
            styleInt = 2  // Grid
        case .image:
            styleInt = 3  // Image (won't use procedural shader)
        }

        return CardBackgroundUniforms(
            color: baseColor,
            lineColor: lineColorSIMD,
            cardSize: SIMD2<Float>(Float(cardSize.x), Float(cardSize.y)),
            lineWidth: Float(lineWidth),
            spacing: Float(spacing),
            marginTop: marginTopPct,
            marginBottom: marginBottomPct,
            marginLeft: marginLeftPct,
            marginRight: marginRightPct,
            style: styleInt,
            imageOpacity: Float(imageOpacity)
        )
    }
}

// MARK: - GPU Uniforms Struct

/// GPU uniforms struct matching Metal shader definition
/// Must match CardBackgroundUniforms in Shaders.metal exactly
struct CardBackgroundUniforms {
    var color: SIMD4<Float>
    var lineColor: SIMD4<Float>
    var cardSize: SIMD2<Float>
    var lineWidth: Float
    var spacing: Float
    var marginTop: Float
    var marginBottom: Float
    var marginLeft: Float
    var marginRight: Float
    var style: Int32
    var imageOpacity: Float
}
