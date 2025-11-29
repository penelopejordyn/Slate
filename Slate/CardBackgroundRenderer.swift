// CardBackgroundRenderer.swift provides utilities to render CardBackgroundView
// to a Metal texture for use in the rendering pipeline.
import SwiftUI
import Metal
import MetalKit
import UIKit

class CardBackgroundRenderer {
    static func renderToTexture(background: CardBackground, size: CGSize, device: MTLDevice) async -> MTLTexture? {
        // Create a UIImage from the SwiftUI view
        guard let uiImage = await renderBackgroundToImage(background: background, size: size) else {
            return nil
        }

        // Convert UIImage to Metal texture
        return createTexture(from: uiImage, device: device)
    }

    @MainActor
    private static func renderBackgroundToImage(background: CardBackground, size: CGSize) -> UIImage? {
        // Create the SwiftUI view
        let view = CardBackgroundView(background: background, size: size)

        // Render to UIImage using ImageRenderer
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2.0 // Use 2x scale for better quality

        // Create UIImage
        return renderer.uiImage
    }

    private static func createTexture(from image: UIImage, device: MTLDevice) -> MTLTexture? {
        guard let cgImage = image.cgImage else { return nil }

        let textureLoader = MTKTextureLoader(device: device)

        do {
            let texture = try textureLoader.newTexture(cgImage: cgImage, options: [
                .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
                .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue)
            ])
            return texture
        } catch {
            print("Failed to create texture from image: \(error)")
            return nil
        }
    }
}

// MARK: - Card Extension

extension Card {
    /// Cached Metal texture for the background
    var backgroundTexture: MTLTexture? {
        get {
            // Use associated objects to store the texture
            return objc_getAssociatedObject(self, &AssociatedKeys.backgroundTexture) as? MTLTexture
        }
        set {
            objc_setAssociatedObject(self, &AssociatedKeys.backgroundTexture, newValue, .OBJC_ASSOCIATION_RETAIN)
        }
    }

    /// Update the background texture if needed
    /// Note: Only called for .image backgrounds now, since .lined and .grid use procedural shaders
    func updateBackgroundTexture(device: MTLDevice) {
        // Only regenerate if we don't have a cached texture or background changed
        // For simplicity, we'll regenerate every time this is called
        // In production, you'd want to cache and only regenerate when background changes

        let textureSize = CGSize(width: size.x, height: size.y)

        switch background.style {
        case .none, .solidColor:
            // No texture needed for these
            backgroundTexture = nil

        case .lined, .grid:
            // These now use procedural shaders, no texture needed
            backgroundTexture = nil

        case .image:
            // Render the background view to a texture
            // This is synchronous for now - we'll load on first access
            Task { @MainActor in
                if let texture = await CardBackgroundRenderer.renderToTexture(
                    background: background,
                    size: textureSize,
                    device: device
                ) {
                    self.backgroundTexture = texture
                }
            }
        }
    }
}

// Associated keys for storing textures
private struct AssociatedKeys {
    static var backgroundTexture: UInt8 = 0
}

// Need to import ObjectiveC for associated objects
import ObjectiveC
