import Foundation
import Metal
import MetalKit

final class PersistenceManager {
    static let shared = PersistenceManager()

    private init() {}

    func exportCanvas(rootFrame: Frame) -> Data? {
        let topFrame = topmostFrame(from: rootFrame)
        let dto = topFrame.toDTO()
        let saveData = CanvasSaveData(timestamp: Date(), rootFrame: dto)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        do {
            return try encoder.encode(saveData)
        } catch {
            print("Export failed: \(error)")
            return nil
        }
    }

    func importCanvas(data: Data, device: MTLDevice) -> Frame? {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        do {
            let saveData = try decoder.decode(CanvasSaveData.self, from: data)
            return restoreFrame(from: saveData.rootFrame, parent: nil, device: device)
        } catch {
            print("Import failed: \(error)")
            return nil
        }
    }

    private func restoreFrame(from dto: FrameDTO, parent: Frame?, device: MTLDevice) -> Frame {
        let frame = Frame(
            id: dto.id,
            parent: parent,
            origin: double2(dto.originInParent),
            scale: dto.scaleRelativeToParent,
            depth: dto.depthFromRoot
        )

        frame.strokes = dto.strokes.map { Stroke(dto: $0, device: device) }

        frame.cards = dto.cards.map { cardDto in
            let type = cardType(from: cardDto.content, device: device)
            let background = cardBackgroundColor(from: cardDto)
            let card = Card(
                id: cardDto.id,
                origin: double2(cardDto.origin),
                size: double2(cardDto.size),
                rotation: cardDto.rotation,
                zoom: cardDto.creationZoom,
                type: type,
                backgroundColor: background,
                opacity: cardDto.opacity ?? 1.0,
                isLocked: cardDto.isLocked ?? false
            )
            card.strokes = cardDto.strokes.map { Stroke(dto: $0, device: device) }
            return card
        }

        frame.children = dto.children.map { restoreFrame(from: $0, parent: frame, device: device) }
        return frame
    }

    private func topmostFrame(from frame: Frame) -> Frame {
        var top = frame
        while let parent = top.parent {
            top = parent
        }
        return top
    }

    private func cardType(from content: CardContentDTO, device: MTLDevice) -> CardType {
        switch content {
        case .solid(let color):
            return .solidColor(float4(color))
        case .lined(let spacing, let lineWidth, let color):
            return .lined(LinedBackgroundConfig(spacing: spacing, lineWidth: lineWidth, color: float4(color)))
        case .grid(let spacing, let lineWidth, let color):
            return .grid(LinedBackgroundConfig(spacing: spacing, lineWidth: lineWidth, color: float4(color)))
        case .image(let data):
            let loader = MTKTextureLoader(device: device)
            if let texture = try? loader.newTexture(data: data, options: [.origin: MTKTextureLoader.Origin.bottomLeft]) {
                return .image(texture)
            }
            return .solidColor(SIMD4<Float>(1, 0, 1, 1))
        }
    }

    private func cardBackgroundColor(from dto: CardDTO) -> SIMD4<Float> {
        if let background = dto.backgroundColor {
            return float4(background)
        }
        switch dto.content {
        case .solid(let color):
            return float4(color)
        default:
            return SIMD4<Float>(1, 1, 1, 1)
        }
    }

    private func double2(_ values: [Double]) -> SIMD2<Double> {
        let x = values.count > 0 ? values[0] : 0
        let y = values.count > 1 ? values[1] : 0
        return SIMD2<Double>(x, y)
    }

    private func float4(_ values: [Float]) -> SIMD4<Float> {
        let x = values.count > 0 ? values[0] : 0
        let y = values.count > 1 ? values[1] : 0
        let z = values.count > 2 ? values[2] : 0
        let w = values.count > 3 ? values[3] : 1
        return SIMD4<Float>(x, y, z, w)
    }
}
