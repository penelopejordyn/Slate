// Card.swift defines the Card model, covering metadata, stroke contents, and size/transform state.
import SwiftUI
import Metal

// Configuration for procedural backgrounds
struct LinedBackgroundConfig {
    var spacing: Float      // Line spacing in points (e.g. 25.0)
    var lineWidth: Float    // Thickness (e.g. 1.0)
    var color: SIMD4<Float> // Line color (e.g. Light Blue)
}

enum CardType {
    case solidColor(SIMD4<Float>) // For testing/backgrounds
    case image(MTLTexture)        // User photos
    case lined(LinedBackgroundConfig) // Procedural Lines
    case grid(LinedBackgroundConfig)  // Procedural Grid
    case drawing([Stroke])        // Future: Mini-canvas inside the card
}

// Helper to get raw values for Metal (Uniform buffer)
struct CardShaderUniforms {
    var spacing: Float
    var lineWidth: Float
    var color: SIMD4<Float>
    var cardWidth: Float  // Needed to calculate UV aspect ratio
    var cardHeight: Float // Needed to calculate UV aspect ratio
}

class Card: Identifiable {
    let id = UUID()

    // MARK: - Physical Properties
    // These are stored in the Parent Frame's coordinate space.
    // They use Double precision to match the Frame's "Local Realism".
    var origin: SIMD2<Double>
    var size: SIMD2<Double>   // Width, Height
    var rotation: Float       // Radians

    // MARK: - Creation Context
    // The "Scale Factor" - Zoom level when this card was created.
    // We use this to calculate how big "25 points" is in world units.
    // This is the "Rosetta Stone" that allows us to translate point sizes
    // into the correct world size forever, regardless of current zoom.
    var creationZoom: Double

    // MARK: - Content
    var type: CardType

    // MARK: - Interaction State
    // When true, the card is selected and can be dragged with finger
    // When false, finger touches pass through to canvas pan
    var isEditing: Bool = false

    // MARK: - Card-Local Strokes
    // Strokes drawn on this card are stored relative to the card's center (0,0)
    // This ensures drawings move with the card when it's repositioned
    var strokes: [Stroke] = []

    // MARK: - Cached Geometry
    // We pre-calculate the 2 triangles (Quad) that represent this card.
    // This optimization means we don't have to do math every frame.
    var localVertices: [StrokeVertex] = []

    init(origin: SIMD2<Double>, size: SIMD2<Double>, rotation: Float = 0, zoom: Double, type: CardType) {
        self.origin = origin
        self.size = size
        self.rotation = rotation
        self.creationZoom = zoom // Store this!
        self.type = type

        // Generate the geometry immediately
        rebuildGeometry()
    }

    // Call this whenever size changes
    func rebuildGeometry() {
        // Create a Quad centered at (0,0) relative to the card's own origin.
        // We will apply the 'origin' translation in the Shader/DrawLoop.
        let halfW = Float(size.x) / 2.0
        let halfH = Float(size.y) / 2.0

        // FIX: Image Inversion
        // Metal textures (Origin BottomLeft) map (0,0) to the bottom-left pixel.
        // We want that pixel at the Bottom-Left of the card.
        // Screen Y is Down. So -H is Top, +H is Bottom.

        // Vertices (Local Space relative to center):
        // TL: (-W, -H) -> Should map to Image Top-Left UV(0, 1)
        // BL: (-W,  H) -> Should map to Image Bottom-Left UV(0, 0)
        // TR: ( W, -H) -> Should map to Image Top-Right UV(1, 1)
        // BR: ( W,  H) -> Should map to Image Bottom-Right UV(1, 0)

        let white = SIMD4<Float>(1, 1, 1, 1) // Cards don't use vertex color
        let v1 = StrokeVertex(position: SIMD2<Float>(-halfW, -halfH), uv: SIMD2<Float>(0, 1), color: white) // Top-Left
        let v2 = StrokeVertex(position: SIMD2<Float>(-halfW,  halfH), uv: SIMD2<Float>(0, 0), color: white) // Bot-Left
        let v3 = StrokeVertex(position: SIMD2<Float>( halfW, -halfH), uv: SIMD2<Float>(1, 1), color: white) // Top-Right
        let v4 = StrokeVertex(position: SIMD2<Float>( halfW,  halfH), uv: SIMD2<Float>(1, 0), color: white) // Bot-Right

        // Triangle Strip or List (List is safer for mixed geometry)
        // Tri 1: TL -> BL -> TR
        // Tri 2: BL -> BR -> TR
        self.localVertices = [v1, v2, v3, v2, v4, v3]
    }

    // MARK: - Hit Testing

    /// Checks if a point (in Parent Frame coordinates) is inside this card's bounds
    /// This handles rotation by inverse-transforming the point into card-local space
    ///
    /// - Parameter pointInFrame: The point to test, in the parent Frame's coordinate system
    /// - Returns: True if the point is inside the card's rectangular bounds
    func hitTest(pointInFrame: SIMD2<Double>) -> Bool {
        // 1. Translate to Card-Local space
        // Move the point so the card's origin is at (0,0)
        let dx = pointInFrame.x - origin.x
        let dy = pointInFrame.y - origin.y

        // 2. Un-rotate (inverse rotation)
        // We rotate the POINT backwards to align with the un-rotated card box
        // This is the inverse of the forward rotation applied in the shader
        let c = cos(-rotation)
        let s = sin(-rotation)

        let localX = dx * Double(c) - dy * Double(s)
        let localY = dx * Double(s) + dy * Double(c)

        // 3. Check bounds (Card is centered at 0,0 in its local space)
        let halfW = size.x / 2.0
        let halfH = size.y / 2.0

        return abs(localX) <= halfW && abs(localY) <= halfH
    }

    // MARK: - Geometry Helpers

    /// Get the 4 corners in Card-Local coordinates (unrotated, relative to center)
    /// This is used for rendering resize handles at constant screen size
    ///
    /// - Returns: Array of 4 corner positions in order: TL, TR, BL, BR
    func getLocalCorners() -> [SIMD2<Float>] {
        let halfW = Float(size.x) / 2.0
        let halfH = Float(size.y) / 2.0
        return [
            SIMD2<Float>(-halfW, -halfH), // Top-Left
            SIMD2<Float>( halfW, -halfH), // Top-Right
            SIMD2<Float>(-halfW,  halfH), // Bottom-Left
            SIMD2<Float>( halfW,  halfH)  // Bottom-Right
        ]
    }
}
