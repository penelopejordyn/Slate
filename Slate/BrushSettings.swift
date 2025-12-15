// BrushSettings.swift stores the current brush configuration (size, color, etc.)
// for use across the drawing system.
import Foundation
import SwiftUI
import simd

class BrushSettings: ObservableObject {
    @Published var size: Double = 10.0  // Stroke width in world units (1-100)
    @Published var color: SIMD4<Float> = SIMD4<Float>(1.0, 0.0, 0.0, 1.0)  // Red default
    @Published var cullingMultiplier: Double = 1.0  // Test multiplier for culling box size (1.0, 0.5, 0.25)

    static let minSize: Double = 1.0
    static let maxSize: Double = 1000.0
}
