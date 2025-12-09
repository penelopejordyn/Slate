// Frame.swift defines the Frame data model used for telescoping reference frames
// that enable hierarchical zooming without precision loss.

//  A "Frame" is a Local Universe - a coordinate system that stays bounded.
//  By chaining Frames together (parent → child), we achieve infinite zoom
//  without ever exceeding Double precision limits.
//

import Foundation
import Metal

/// A Frame represents a bounded coordinate system (a "Local Universe").
///
/// **The Philosophy:**
/// Instead of one infinite coordinate system that breaks at 10^15, we use
/// a linked list of finite coordinate systems. When zoom exceeds a threshold,
/// we create a new Frame inside the current one and reset the zoom to 1.0.
///
/// **The Analogy:**
/// Like the movie "Men in Black" - the galaxy is inside a marble, which is
/// inside a bag, which is inside a locker. Each level is a separate Frame.
class Frame: Identifiable {
    let id = UUID()

    /// Strokes that belong to this specific "depth" or layer of reality
    var strokes: [Stroke] = []

    /// Cards that belong to this Frame (Images, PDFs, Sketches)
    var cards: [Card] = []

    /// The "Universe" containing this frame (nil for root frame)
    var parent: Frame?

    /// Where this frame's center lives inside the Parent's coordinate system
    /// (Used to render this frame when the camera is in the Parent)
    var originInParent: SIMD2<Double>

    /// How much smaller this frame is compared to the Parent
    /// Example: If we drilled down at 1000× zoom, this would be 1000.0
    /// This tells us: "1 unit in parent space = 1000 units in this frame"
    var scaleRelativeToParent: Double

    /// Track the sub-universes created inside this frame
    /// This allows us to re-enter existing frames instead of creating parallel universes
    var children: [Frame] = []

    // MARK: - Tile Caching (Canvas Baking)

    /// Tile key in world space. Level is reserved for future LOD tiers.
    struct TileKey: Hashable {
        let x: Int
        let y: Int
        let level: Int
    }

    /// Cached tile texture and dirtiness flag.
    struct Tile {
        var texture: MTLTexture?
        var dirty: Bool
        let key: TileKey
        let worldRect: CGRect
    }

    /// Fixed world-space tile size (width/height in frame units).
    static let tileWorldSize: Double = 1024.0

    /// Fixed resolution used when baking to textures.
    static let tileTextureSize: Int = 512

    /// Sparse grid of baked tiles keyed by integer tile coordinates.
    var tiles: [TileKey: Tile] = [:]

    /// Initialize a new Frame
    /// - Parameters:
    ///   - parent: The containing universe (nil for root)
    ///   - origin: Where this frame lives in parent coordinates
    ///   - scale: The zoom factor when this frame was created
    init(parent: Frame? = nil, origin: SIMD2<Double> = .zero, scale: Double = 1.0) {
        self.parent = parent
        self.originInParent = origin
        self.scaleRelativeToParent = scale
    }

    // MARK: - Tile Helpers

    /// Calculate the world rect for a given tile key.
    func worldRect(for key: TileKey) -> CGRect {
        let origin = CGPoint(
            x: Double(key.x) * Frame.tileWorldSize,
            y: Double(key.y) * Frame.tileWorldSize
        )
        let size = CGSize(width: Frame.tileWorldSize, height: Frame.tileWorldSize)
        return CGRect(origin: origin, size: size)
    }

    /// Get the tile key for a world position.
    func tileKey(for position: SIMD2<Double>, level: Int = 0) -> TileKey {
        let x = Int(floor(position.x / Frame.tileWorldSize))
        let y = Int(floor(position.y / Frame.tileWorldSize))
        return TileKey(x: x, y: y, level: level)
    }

    /// Find all tile keys that overlap the provided world rect.
    func tileKeys(overlapping rect: CGRect, level: Int = 0) -> [TileKey] {
        guard rect.width > 0, rect.height > 0 else { return [] }
        let minX = Int(floor(rect.minX / Frame.tileWorldSize))
        let maxX = Int(floor(rect.maxX / Frame.tileWorldSize))
        let minY = Int(floor(rect.minY / Frame.tileWorldSize))
        let maxY = Int(floor(rect.maxY / Frame.tileWorldSize))

        var keys: [TileKey] = []
        for x in minX...maxX {
            for y in minY...maxY {
                keys.append(TileKey(x: x, y: y, level: level))
            }
        }
        return keys
    }

    /// Mark tiles intersecting the stroke as dirty so they are re-baked.
    func markTilesDirty(for stroke: Stroke) {
        let bounds = stroke.localBounds
        let rect = CGRect(
            x: stroke.origin.x + Double(bounds.origin.x),
            y: stroke.origin.y + Double(bounds.origin.y),
            width: Double(bounds.width),
            height: Double(bounds.height)
        )

        for key in tileKeys(overlapping: rect) {
            let tileRect = worldRect(for: key)
            let existing = tiles[key]
            let newTile = Tile(
                texture: existing?.texture,
                dirty: true,
                key: key,
                worldRect: tileRect
            )
            tiles[key] = newTile
        }
    }
}
