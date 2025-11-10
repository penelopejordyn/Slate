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
    
    // Cache vertices by zoom threshold
    private var verticesCache: [Int: [SIMD2<Float>]] = [:]
    private let viewSize: CGSize
    
    init(centerPoints: [CGPoint], width: CGFloat, color: SIMD4<Float>, viewSize: CGSize) {
        self.id = UUID()
        self.centerPoints = centerPoints
        self.width = width
        self.color = color
        self.viewSize = viewSize
    }
    
    // Get vertices for current zoom, tessellating if needed
    mutating func vertices(for zoom: Float) -> [SIMD2<Float>] {
        let zoomThreshold = zoomThresholdForZoom(zoom)
        
        // Return cached if available
        if let cached = verticesCache[zoomThreshold] {
            return cached
        }
        
        // Otherwise tessellate and cache
        let segments = segmentsForZoom(zoom)
        let vertices = tessellateStroke(
            centerPoints: centerPoints,
            width: width,
            viewSize: viewSize,
            panOffset: .zero,
            zoomScale: 1.0,
            segmentsPerCurve: segments
        )
        verticesCache[zoomThreshold] = vertices
        return vertices
    }
    
    // Calculate which threshold this zoom level falls into
    private func zoomThresholdForZoom(_ zoom: Float) -> Int {
        return Int(floor(zoom / 5000.0)) * 5000
    }
    
    // Calculate segments: double every 5000x zoom
    private func segmentsForZoom(_ zoom: Float) -> Int {
        let baseSegments = 20
        let doublings = Int(floor(zoom / 5000.0))
        
        // Cap at 320 segments to prevent explosion at extreme zooms
        let segments = baseSegments * Int(pow(2.0, Double(min(doublings, 4))))
        return min(segments, 320)
    }
}
