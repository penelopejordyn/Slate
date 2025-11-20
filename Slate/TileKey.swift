//
//  TileKey.swift
//  Slate
//
//  Created by Penny Marshall on 11/13/25.
//

import Foundation

/// Represents a unique tile address using a simple grid-based system.
/// At zoom level L, tiles are 1024 / 2^L pixels wide in world space.
/// Each tile is identified by its level and grid coordinates (tx, ty).
struct TileKey: Hashable {
    /// Zoom level: floor(log2(zoomScale))
    /// At level L, tile size in world = 1024 / 2^L pixels
    let level: Int32

    /// Tile X coordinate in grid at this level
    /// tileOriginX = tx * tileSizeWorld
    let tx: Int64

    /// Tile Y coordinate in grid at this level
    /// tileOriginY = ty * tileSizeWorld
    let ty: Int64

    /// Human-readable description
    var description: String {
        return "Level \(level): (\(tx), \(ty))"
    }
}
