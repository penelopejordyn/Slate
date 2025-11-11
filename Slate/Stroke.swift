//
//  Stroke.swift
//  Slate
//
//  Created by Penny Marshall on 10/28/25.
//

import SwiftUI

struct Stroke {
    let id: UUID
    let centerPoints: [CGPoint]
    let width: CGFloat
    let color: SIMD4<Float>

    /// Cache of tile-local triangles per zoom level.
    private(set) var tileCache: [Int32: [TileKey: [SIMD2<Float>]]] = [:]

    init(centerPoints: [CGPoint], width: CGFloat, color: SIMD4<Float>) {
        self.id = UUID()
        self.centerPoints = centerPoints
        self.width = width
        self.color = color
        self.tileCache = [:]
    }

    mutating func tiles(for level: Int32, tileSize: Double) -> [TileKey: [SIMD2<Float>]] {
        if let cached = tileCache[level] {
            return cached
        }

        let computed = tessellateStrokeIntoTiles(centerPoints: centerPoints,
                                                 width: width,
                                                 tileSize: tileSize)
        tileCache[level] = computed

        if tileCache.count > 8 {
            let sortedKeys = tileCache.keys.sorted { lhs, rhs in
                let dl = abs(Int(lhs) - Int(level))
                let dr = abs(Int(rhs) - Int(level))
                if dl == dr { return lhs < rhs }
                return dl < dr
            }
            for key in sortedKeys.dropFirst(8) {
                tileCache.removeValue(forKey: key)
            }
        }

        return computed
    }
}
