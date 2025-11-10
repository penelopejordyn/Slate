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
    let vertices: [SIMD2<Double>]  // ← Pre-tessellated at identity (double precision)
    
    init(centerPoints: [CGPoint], width: CGFloat, color: SIMD4<Float>, viewSize: CGSize) {
        self.id = UUID()
        self.centerPoints = centerPoints
        self.width = width
        self.color = color
        
        // Tessellate ONCE at identity transform
        self.vertices = tessellateStroke(
            centerPoints: centerPoints,
            width: width,
            viewSize: viewSize,
            panOffset: .zero,    // ← Identity
            zoomScale: 1.0
        )
    }
}
