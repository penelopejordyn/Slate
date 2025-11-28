// TileManager.swift manages the floating reference tiling grid used for infinite zoom and pan.
import Foundation
import CoreGraphics

/// Manages the floating reference tiling system for infinite zoom/pan.
/// The reference level automatically adjusts based on zoom, keeping tile paths short.
class TileManager {
    /// Base tile size in pixels at reference level 0
    static let baseTileSize: Double = 1024.0

    // MARK: - Debug Mode

    /// Enable/disable debug output (toggle with long-press)
    var debugMode: Bool = false

    // MARK: - Current Transform State

    /// Current zoom scale (updates every frame)
    var currentZoomScale: Float = 1.0

    /// Current pan offset in pixels (updates every frame)
    var currentPanOffset: SIMD2<Float> = .zero

    /// Current view size (updates when view size changes)
    var viewSize: CGSize = .zero

    // MARK: - Computed Properties

    /// Calculate view center in world coordinates.
    /// This is the point that should have the shortest tile path.
    var viewCenterInWorld: CGPoint {
        // View center in screen coordinates
        let centerX = viewSize.width / 2
        let centerY = viewSize.height / 2

        // In the simplified case (no rotation, no zoom transform applied yet):
        // The view center in world coordinates is just the screen center
        // minus the pan offset (since pan moves the world)
        // Note: This is an approximation. For exact inverse transform,
        // we'd need rotation angle, but for tile addressing this is sufficient.
        let worldX = centerX - CGFloat(currentPanOffset.x)
        let worldY = centerY - CGFloat(currentPanOffset.y)

        return CGPoint(x: worldX, y: worldY)
    }

    /// Reference origin - the origin point of the reference tile in world coordinates.
    /// This "floats" with your view, keeping tile paths short.
    var referenceOrigin: (x: Double, y: Double) {
        let tileSize = Self.baseTileSize
        let center = viewCenterInWorld

        // Snap view center to the nearest tile boundary at reference level
        // This creates a stable reference tile that contains or is near the view center
        let tileX = floor(Double(center.x) / tileSize)
        let tileY = floor(Double(center.y) / tileSize)

        return (x: tileX * tileSize, y: tileY * tileSize)
    }

    /// Reference level based on current zoom.
    /// floor(log2(zoomScale)) ensures we work at appropriate granularity.
    var referenceLevel: Int {
        if currentZoomScale <= 0 { return 0 }
        return Int(floor(log2(Double(currentZoomScale))))
    }

    /// Zoom mantissa - the "fractional" part between power-of-2 levels.
    /// Always in range [1.0, 2.0)
    var zoomMantissa: Double {
        let level = referenceLevel
        return Double(currentZoomScale) / pow(2.0, Double(level))
    }

    /// Tile size at current reference level in world pixels.
    /// At level L, tiles are 1024 / 2^L world pixels wide.
    var tileSizeAtReferenceLevel: Double {
        return Self.baseTileSize / pow(2.0, Double(referenceLevel))
    }

    /// Get tile size in world pixels for a specific level
    func tileSizeForLevel(_ level: Int32) -> Double {
        return Self.baseTileSize / pow(2.0, Double(level))
    }

    // MARK: - Width Conversion

    /// Convert world-space width to tile-local width.
    ///
    /// Example at zoom 1,000,000x:
    /// - widthWorld = 10.0 / 1,000,000 = 0.00001 world pixels
    /// - level = 20, tileSize = 1024 / 2^20 = 0.000976 world pixels
    /// - widthTile = 0.00001 / 0.000976 = 0.0102 tile units 
    ///
    /// This keeps width in a stable range (~0.01) regardless of zoom!
    ///
    /// - Parameter worldWidth: Width in world pixels
    /// - Returns: Width in tile-local units (relative to 1024.0 tile size)
    func worldWidthToTileWidth(_ worldWidth: Double) -> Double {
        let tileSizeWorld = tileSizeAtReferenceLevel
        guard tileSizeWorld > 0 else { return worldWidth }
        return worldWidth / tileSizeWorld
    }

    /// Calculate tile-local width for a stroke at current zoom level.
    /// This is the standard width calculation.
    ///
    /// - Parameter baseWidth: Base width in screen pixels (e.g., 10.0)
    /// - Returns: Width in tile-local units
    func calculateTileLocalWidth(baseWidth: Double = 10.0) -> Double {
        // World width shrinks with zoom (to maintain constant screen width)
        let worldWidth = baseWidth / Double(currentZoomScale)

        // Convert to tile-local width
        return worldWidthToTileWidth(worldWidth)
    }

    // MARK: - Coordinate Conversion

    /// Convert a world point to tile-local coordinates in [0, 1024].
    ///
    /// This is the CRITICAL function that keeps coordinates small!
    ///
    /// - Parameters:
    ///   - worldPoint: Point in world coordinates
    ///   - tileKey: The tile to convert relative to
    /// - Returns: Point in tile-local space [0, 1024]
    func worldToTileLocal(worldPoint: CGPoint, tileKey: TileKey) -> CGPoint {
        let tileSizeWorld = tileSizeForLevel(tileKey.level)

        // Tile origin in world coordinates
        let tileOriginX = Double(tileKey.tx) * tileSizeWorld
        let tileOriginY = Double(tileKey.ty) * tileSizeWorld

        // Point relative to tile origin (in world pixels)
        let localWorldX = Double(worldPoint.x) - tileOriginX
        let localWorldY = Double(worldPoint.y) - tileOriginY

        // Normalize to [0, 1024] for tessellation
        // This is the magic that keeps numbers small!
        let localX = (localWorldX / tileSizeWorld) * Self.baseTileSize
        let localY = (localWorldY / tileSizeWorld) * Self.baseTileSize

        return CGPoint(x: localX, y: localY)
    }

    /// Convert a tile-local point back to world coordinates.
    ///
    /// - Parameters:
    ///   - localPoint: Point in tile-local space [0, 1024]
    ///   - tileKey: The tile this point is relative to
    /// - Returns: Point in world coordinates
    func tileLocalToWorld(localPoint: CGPoint, tileKey: TileKey) -> CGPoint {
        let tileSizeWorld = tileSizeForLevel(tileKey.level)

        // Tile origin in world coordinates
        let tileOriginX = Double(tileKey.tx) * tileSizeWorld
        let tileOriginY = Double(tileKey.ty) * tileSizeWorld

        // Denormalize from [0, 1024] to [0, tileSizeWorld]
        let localWorldX = (Double(localPoint.x) / Self.baseTileSize) * tileSizeWorld
        let localWorldY = (Double(localPoint.y) / Self.baseTileSize) * tileSizeWorld

        // Add tile origin to get world coordinates
        let worldX = tileOriginX + localWorldX
        let worldY = tileOriginY + localWorldY

        return CGPoint(x: worldX, y: worldY)
    }

    /// Calculate the world-space origin of a tile given its TileKey.
    ///
    /// - Parameter tileKey: The tile address
    /// - Returns: The tile's bottom-left corner in world coordinates
    func tileOriginInWorld(tileKey: TileKey) -> (x: Double, y: Double) {
        let tileSizeWorld = tileSizeForLevel(tileKey.level)
        return (x: Double(tileKey.tx) * tileSizeWorld,
                y: Double(tileKey.ty) * tileSizeWorld)
    }


    // MARK: - Core Algorithm

    /// Convert a world point to a TileKey using simple grid-based tiling.
    /// This is the CORRECT approach that keeps tile-local coordinates small.
    ///
    /// Algorithm (simple grid-based):
    /// 1. Use current reference level
    /// 2. Calculate tile size in world: 1024 / 2^level
    /// 3. Find tile grid coordinates: (floor(wx / tileSize), floor(wy / tileSize))
    ///
    /// This ensures tiles at level L are exactly tileSizeWorld pixels wide,
    /// and tile-local coordinates are always in [0, tileSizeWorld), which
    /// normalizes to [0, 1024] for tessellation.
    ///
    /// - Parameter worldPoint: Point in world coordinates (canvas pixels)
    /// - Returns: TileKey with level and grid coordinates
    func getTileKey(worldPoint: CGPoint) -> TileKey {
        let level = Int32(referenceLevel)
        let tileSizeWorld = tileSizeForLevel(level)

        // Find which tile this point falls into
        let tx = Int64(floor(Double(worldPoint.x) / tileSizeWorld))
        let ty = Int64(floor(Double(worldPoint.y) / tileSizeWorld))

        return TileKey(level: level, tx: tx, ty: ty)
    }

    /// Format debug information for a touch event
    func debugInfo(worldPoint: CGPoint, screenPoint: CGPoint, tileKey: TileKey) -> String {
        let localPoint = worldToTileLocal(worldPoint: worldPoint, tileKey: tileKey)
        let tileOrigin = tileOriginInWorld(tileKey: tileKey)
        let tileSizeWorld = tileSizeForLevel(tileKey.level)

        var output = """
         TOUCH EVENT (Grid-Based Tiling)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Zoom State:
          Scale: \(currentZoomScale)x
          Reference Level: \(referenceLevel)
          Tile Size (world): \(String(format: "%.9f", tileSizeWorld)) px

        Touch → World:
          Screen: (\(String(format: "%.1f", screenPoint.x)), \(String(format: "%.1f", screenPoint.y)))
          World: (\(String(format: "%.3f", worldPoint.x)), \(String(format: "%.3f", worldPoint.y)))

        Tile Address (Grid):
          Level: \(tileKey.level)
          Grid (tx, ty): (\(tileKey.tx), \(tileKey.ty))
          Tile Origin (world): (\(String(format: "%.6f", tileOrigin.x)), \(String(format: "%.6f", tileOrigin.y)))
          Tile Size (world): \(String(format: "%.9f", tileSizeWorld)) px

        Tile-Local Coordinates:
          Local [0, 1024]: (\(String(format: "%.3f", localPoint.x)), \(String(format: "%.3f", localPoint.y)))
        """

        // Check if coordinates are in valid range
        if localPoint.x < 0 || localPoint.x > 1024 || localPoint.y < 0 || localPoint.y > 1024 {
            output += "\n   WARNING: OUT OF RANGE!"
        } else {
            output += "\n   Coordinates in valid range [0, 1024]"
        }

        output += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        return output
    }
}


