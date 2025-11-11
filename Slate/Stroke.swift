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

    init(centerPoints: [CGPoint], width: CGFloat, color: SIMD4<Float>) {
        self.id = UUID()
        self.centerPoints = centerPoints
        self.width = width
        self.color = color
    }
}
