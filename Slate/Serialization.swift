import Foundation

// MARK: - Save File Schema

struct CanvasSaveData: Codable {
    let version: Int
    let timestamp: Date
    let rootFrame: FrameDTO

    init(timestamp: Date, rootFrame: FrameDTO, version: Int = 1) {
        self.version = version
        self.timestamp = timestamp
        self.rootFrame = rootFrame
    }
}

// MARK: - DTOs

struct FrameDTO: Codable {
    let id: UUID
    let originInParent: [Double]
    let scaleRelativeToParent: Double
    let depthFromRoot: Int
    let strokes: [StrokeDTO]
    let cards: [CardDTO]
    let children: [FrameDTO]
}

struct StrokeDTO: Codable {
    let id: UUID
    let origin: [Double]
    let worldWidth: Double
    let color: [Float]
    let zoomCreation: Float
    let depthID: UInt32
    let depthWrite: Bool
    let points: [[Float]]
}

struct CardDTO: Codable {
    let id: UUID
    let origin: [Double]
    let size: [Double]
    let rotation: Float
    let creationZoom: Double
    let content: CardContentDTO
    let strokes: [StrokeDTO]
    let backgroundColor: [Float]?
    let opacity: Float?
    let isLocked: Bool?
}

enum CardContentDTO: Codable {
    case solid(color: [Float])
    case image(pngData: Data)
    case lined(spacing: Float, lineWidth: Float, color: [Float])
    case grid(spacing: Float, lineWidth: Float, color: [Float])
}
