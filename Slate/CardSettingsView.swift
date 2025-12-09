// CardSettingsView.swift provides the UI for editing card properties (type, background, etc.)
import SwiftUI
import MetalKit

struct CardSettingsView: View {
    let card: Card // Reference class, so modifying it updates Metal immediately

    @State private var selectedTab = 0
    @State private var spacing: Float = 25.0
    @State private var lineWidth: Float = 1.0
    @State private var showImagePicker = false
    @State private var uiImage: UIImage?
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
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

                Spacer()
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
            // Initialize state from current card type
            switch card.type {
            case .solidColor:
                selectedTab = 0
            case .lined(let config):
                selectedTab = 1
                spacing = config.spacing
                lineWidth = config.lineWidth
            case .grid(let config):
                selectedTab = 2
                spacing = config.spacing
                lineWidth = config.lineWidth
            case .image:
                selectedTab = 3
            case .drawing:
                selectedTab = 0 // Default to solid for now
            }
        }
    }

    /// Update the card type based on selected tab and current settings
    private func updateCardType(for tab: Int) {
        switch tab {
        case 0: // Solid
            card.type = .solidColor(SIMD4<Float>(1.0, 1.0, 1.0, 1.0)) // White
        case 1: // Lined
            card.type = .lined(LinedBackgroundConfig(
                spacing: spacing,
                lineWidth: lineWidth,
                color: SIMD4<Float>(0.7, 0.8, 1.0, 0.5) // Light blue
            ))
        case 2: // Grid
            card.type = .grid(LinedBackgroundConfig(
                spacing: spacing,
                lineWidth: lineWidth,
                color: SIMD4<Float>(0.7, 0.8, 1.0, 0.5) // Light blue
            ))
        case 3: // Image
            // Will be handled in Phase 4
            break
        default:
            break
        }
    }
}
