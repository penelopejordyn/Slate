// CardSettingsView.swift provides the UI for editing card properties (type, background, etc.)
import SwiftUI
import MetalKit
import UIKit

struct CardSettingsView: View {
    let card: Card // Reference class, so modifying it updates Metal immediately
    let onDelete: (() -> Void)?

    @State private var selectedTab = 0
    @State private var spacing: Float = 25.0
    @State private var lineWidth: Float = 1.0
    @State private var backgroundColor: Color = .white
    @State private var lineColor: Color = Color(.sRGB, red: 0.7, green: 0.8, blue: 1.0, opacity: 0.5)
    @State private var cardOpacity: Float = 1.0
    @State private var isLocked: Bool = false
    @State private var showImagePicker = false
    @State private var uiImage: UIImage?
    @Environment(\.dismiss) var dismiss

    init(card: Card, onDelete: (() -> Void)? = nil) {
        self.card = card
        self.onDelete = onDelete
    }

    private var backgroundBinding: Binding<Color> {
        Binding(
            get: { backgroundColor },
            set: { newColor in
                backgroundColor = newColor
                applyBackgroundColor(newColor)
            }
        )
    }

    private var lineColorBinding: Binding<Color> {
        Binding(
            get: { lineColor },
            set: { newColor in
                lineColor = newColor
                updateCardType(for: selectedTab)
            }
        )
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Type Picker
                    Picker("Background Type", selection: $selectedTab) {
                        Text("Solid").tag(0)
                        Text("Lined").tag(1)
                        Text("Grid").tag(2)
                        Text("Image").tag(3)
                    }
                    .pickerStyle(.segmented)
                    .padding()
                    .onChange(of: selectedTab) { _, newValue in
                        updateCardType(for: newValue)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Background Color")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        ColorPicker("Background Color", selection: backgroundBinding, supportsOpacity: false)
                            .font(.subheadline)
                    }
                    .padding(.horizontal)

                    // Opacity Slider (appears on all tabs)
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Opacity: \(Int(cardOpacity * 100))%")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Slider(value: $cardOpacity, in: 0.0...1.0, step: 0.05)
                            .onChange(of: cardOpacity) { _, newValue in
                                card.opacity = newValue
                            }
                    }
                    .padding(.horizontal)

                    VStack(alignment: .leading, spacing: 12) {
                        Toggle(isOn: Binding(
                            get: { isLocked },
                            set: { newValue in
                                isLocked = newValue
                                card.isLocked = newValue
                                if newValue {
                                    card.isEditing = false
                                }
                            }
                        )) {
                            HStack(spacing: 8) {
                                Image(systemName: isLocked ? "lock.fill" : "lock.open")
                                Text("Lock Card")
                            }
                            .font(.subheadline)
                        }

                        Button(role: .destructive) {
                            onDelete?()
                            dismiss()
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "trash")
                                Text("Delete Card")
                            }
                            .font(.subheadline)
                        }
                    }
                    .padding(.horizontal)

                    // Settings for selected type
                    if selectedTab == 1 || selectedTab == 2 {
                        // Lined or Grid settings
                        VStack(alignment: .leading, spacing: 16) {
                            // Line Spacing Slider
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Line Spacing: \(Int(spacing)) pt")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Slider(value: $spacing, in: 10...100, step: 5)
                                    .onChange(of: spacing) {
                                        updateCardType(for: selectedTab)
                                    }
                            }

                            VStack(alignment: .leading, spacing: 8) {
                                Text("Line Color")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                ColorPicker("Line Color", selection: lineColorBinding, supportsOpacity: true)
                                    .font(.subheadline)
                            }

                            // Line Width Slider
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Line Width: \(String(format: "%.1f", lineWidth)) pt")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Slider(value: $lineWidth, in: 0.5...5.0, step: 0.5)
                                    .onChange(of: lineWidth) {
                                        updateCardType(for: selectedTab)
                                    }
                            }
                        }
                        .padding(.horizontal)

                    } else if selectedTab == 0 {
                        // Solid color settings
                        Text("Solid color background")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .padding()

                    } else if selectedTab == 3 {
                        // Image settings
                        VStack(spacing: 16) {
                            Button(action: {
                                showImagePicker = true
                            }) {
                                HStack {
                                    Image(systemName: "photo.on.rectangle")
                                    Text("Select Image")
                                }
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding()
                                .frame(maxWidth: .infinity)
                                .background(Color.blue)
                                .cornerRadius(12)
                            }

                            if case .image = card.type {
                                Text("Image loaded")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(.horizontal)
                    }

                    Spacer(minLength: 20)
                }
                .padding(.vertical, 12)
            }
            .navigationTitle("Card Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        .sheet(isPresented: $showImagePicker) {
            ImagePicker(image: $uiImage)
        }
        .onChange(of: uiImage) { _, newImage in
            guard let img = newImage else { return }

            // Load Texture
            let device = MTLCreateSystemDefaultDevice()!
            let loader = MTKTextureLoader(device: device)

            if let cgImg = img.cgImage,
               let texture = try? loader.newTexture(cgImage: cgImg, options: [.origin: MTKTextureLoader.Origin.bottomLeft]) {

                card.type = .image(texture)

                // Optional: Resize card to match image aspect ratio
                let aspect = Double(img.size.width / img.size.height)
                let newHeight = card.size.x / aspect
                card.size.y = newHeight
                card.rebuildGeometry() // Important! Rebuild quad vertices
            }
        }
        .onAppear {
            // Initialize state from current card
            backgroundColor = colorFromSIMD(card.backgroundColor)
            cardOpacity = card.opacity
            isLocked = card.isLocked
            switch card.type {
            case .solidColor:
                selectedTab = 0
            case .lined(let config):
                selectedTab = 1
                spacing = config.spacing
                lineWidth = config.lineWidth
                lineColor = colorFromSIMD(config.color)
            case .grid(let config):
                selectedTab = 2
                spacing = config.spacing
                lineWidth = config.lineWidth
                lineColor = colorFromSIMD(config.color)
            case .image:
                selectedTab = 3
            case .drawing:
                selectedTab = 0 // Default to solid for now
            }
        }
    }

    /// Update the card type based on selected tab and current settings
    private func updateCardType(for tab: Int) {
        let background = simdFromColor(backgroundColor)
        card.backgroundColor = background

        switch tab {
        case 0: // Solid
            card.type = .solidColor(background)
        case 1: // Lined
            card.type = .lined(LinedBackgroundConfig(
                spacing: spacing,
                lineWidth: lineWidth,
                color: simdFromColor(lineColor)
            ))
        case 2: // Grid
            card.type = .grid(LinedBackgroundConfig(
                spacing: spacing,
                lineWidth: lineWidth,
                color: simdFromColor(lineColor)
            ))
        case 3: // Image
            // Will be handled in Phase 4
            break
        default:
            break
        }
    }

    private func applyBackgroundColor(_ color: Color) {
        let background = simdFromColor(color)
        card.backgroundColor = background
        if case .solidColor = card.type {
            card.type = .solidColor(background)
        }
    }

    private func simdFromColor(_ color: Color) -> SIMD4<Float> {
        let uiColor = UIColor(color)
        var r: CGFloat = 1
        var g: CGFloat = 1
        var b: CGFloat = 1
        var a: CGFloat = 1
        if uiColor.getRed(&r, green: &g, blue: &b, alpha: &a) {
            return SIMD4<Float>(Float(r), Float(g), Float(b), Float(a))
        }
        return SIMD4<Float>(1, 1, 1, 1)
    }

    private func colorFromSIMD(_ color: SIMD4<Float>) -> Color {
        Color(.sRGB,
              red: Double(color.x),
              green: Double(color.y),
              blue: Double(color.z),
              opacity: Double(color.w))
    }
}
