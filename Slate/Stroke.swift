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
    let vertices: [SIMD2<Float>]  // ← Cached in world space (no transform)
    
    init(centerPoints: [CGPoint], width: CGFloat, color: SIMD4<Float>, viewSize: CGSize) {
        self.id = UUID()
        self.centerPoints = centerPoints
        self.width = width
        self.color = color
        
        // Tessellate ONCE with no transform
        self.vertices = tessellateStroke(
            centerPoints: centerPoints,
            width: width,
            viewSize: viewSize,
            panOffset: .zero,      // ← No transform!
            zoomScale: 1.0
        )
    }
}
