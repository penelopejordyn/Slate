// CatmullRom.swift implements Catmull-Rom spline sampling utilities for smooth stroke interpolation.
import Foundation
import CoreGraphics

// MARK: - Main Catmull-Rom Function

func catmullRomPoints(points: [CGPoint],
                     closed: Bool = false,
                     alpha: CGFloat = 0.5,
                     segmentsPerCurve: Int = 20) -> [CGPoint] {
    guard points.count >= 4 else { return points }
    
    var result: [CGPoint] = []
    
    let startIndex = closed ? 0 : 1
    let endIndex = closed ? points.count : points.count - 2
    
    for i in startIndex..<endIndex {
        let p0 = points[i-1 < 0 ? points.count - 1 : i - 1]
        let p1 = points[i]
        let p2 = points[(i+1) % points.count]
        let p3Index = (i+1) % points.count + 1
        let p3 = points[p3Index >= points.count ? p3Index - points.count : p3Index]
        
        // Calculate distances
        let d1 = distance(p1, p0)
        let d2 = distance(p2, p1)
        let d3 = distance(p3, p2)
        
        // Calculate control points (Catmull-Rom to Bezier conversion)
        var b1 = multiply(p2, pow(d1, 2 * alpha))
        b1 = subtract(b1, multiply(p0, pow(d2, 2 * alpha)))
        b1 = add(b1, multiply(p1, 2 * pow(d1, 2 * alpha) + 3 * pow(d1, alpha) * pow(d2, alpha) + pow(d2, 2 * alpha)))
        b1 = multiply(b1, 1.0 / (3 * pow(d1, alpha) * (pow(d1, alpha) + pow(d2, alpha))))
        
        var b2 = multiply(p1, pow(d3, 2 * alpha))
        b2 = subtract(b2, multiply(p3, pow(d2, 2 * alpha)))
        b2 = add(b2, multiply(p2, 2 * pow(d3, 2 * alpha) + 3 * pow(d3, alpha) * pow(d2, alpha) + pow(d2, 2 * alpha)))
        b2 = multiply(b2, 1.0 / (3 * pow(d3, alpha) * (pow(d3, alpha) + pow(d2, alpha))))
        
        // Add start point only on first segment
        if i == startIndex {
            result.append(p1)
        }

        //  OPTIMIZATION: Adaptive Segments
        // Calculate the physical distance of this segment
        let segmentDistance = distance(p1, p2)

        // Rule: 1 segment per 2 pixels of length
        // This prevents vertex explosion for slow strokes where points are close together
        // Min: 1 segment (don't generate zero segments)
        // Max: 20 segments (don't over-smooth long lines)
        let estimatedSegments = Int(segmentDistance / 2.0)
        let adaptiveSegments = min(max(1, estimatedSegments), 20)

        // Subdivide using the adaptive segment count
        for j in 1...adaptiveSegments {
            let t = CGFloat(j) / CGFloat(adaptiveSegments)
            let point = cubicBezierPoint(p0: p1, p1: b1, p2: b2, p3: p2, t: t)
            result.append(point)
        }
    }
    
    return result
}

// MARK: - Bezier Evaluation

func cubicBezierPoint(p0: CGPoint, p1: CGPoint, p2: CGPoint, p3: CGPoint, t: CGFloat) -> CGPoint {
    let oneMinusT = 1.0 - t
    let oneMinusT2 = oneMinusT * oneMinusT
    let oneMinusT3 = oneMinusT2 * oneMinusT
    let t2 = t * t
    let t3 = t2 * t
    
    let x = oneMinusT3 * p0.x +
            3 * oneMinusT2 * t * p1.x +
            3 * oneMinusT * t2 * p2.x +
            t3 * p3.x
    
    let y = oneMinusT3 * p0.y +
            3 * oneMinusT2 * t * p1.y +
            3 * oneMinusT * t2 * p2.y +
            t3 * p3.y
    
    return CGPoint(x: x, y: y)
}

// MARK: - Helper Functions

func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
    let dx = p2.x - p1.x
    let dy = p2.y - p1.y
    return sqrt(dx * dx + dy * dy)
}

func multiply(_ p: CGPoint, _ value: CGFloat) -> CGPoint {
    return CGPoint(x: p.x * value, y: p.y * value)
}

func add(_ p1: CGPoint, _ p2: CGPoint) -> CGPoint {
    return CGPoint(x: p1.x + p2.x, y: p1.y + p2.y)
}

func subtract(_ p1: CGPoint, _ p2: CGPoint) -> CGPoint {
    return CGPoint(x: p1.x - p2.x, y: p1.y - p2.y)
}
