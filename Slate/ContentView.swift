// ContentView.swift wires the SwiftUI interface, hosting the toolbar and MetalView canvas binding.
import SwiftUI

struct ContentView: View {
    // Reference to MetalView's coordinator for adding cards
    @State private var metalViewCoordinator: MetalView.Coordinator?
    @State private var editingCard: Card? // The card being edited
    @State private var showSettingsSheet = false

    var body: some View {
        ZStack(alignment: .topTrailing) {
            MetalView(coordinator: $metalViewCoordinator)
            .edgesIgnoringSafeArea(.all)
            .onChange(of: metalViewCoordinator) { _, newCoord in
                // Bind the callback when coordinator is set
                newCoord?.onEditCard = { card in
                    self.editingCard = card
                    self.showSettingsSheet = true
                }
            }

            VStack(spacing: 16) {
                // Add Card Button
                Button(action: {
                    metalViewCoordinator?.addCard()
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: "plus.rectangle.fill")
                        Text("Add Card")
                    }
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(.ultraThinMaterial)
                    .cornerRadius(20)
                }

                // Stroke Size Slider
                if let coordinator = metalViewCoordinator {
                    StrokeSizeSlider(brushSettings: coordinator.brushSettings)
                        .frame(width: 200)
                }
            }
            .padding(.top, 60)
            .padding(.trailing, 16)
        }
        .sheet(isPresented: $showSettingsSheet) {
            if let card = editingCard {
                CardSettingsView(card: card)
                    .presentationDetents([.medium])
            }
        }
    }
}

// MARK: - Stroke Size Slider Component

struct StrokeSizeSlider: View {
    @ObservedObject var brushSettings: BrushSettings

    // Convert SIMD4<Float> to SwiftUI Color
    private var strokeColor: Binding<Color> {
        Binding(
            get: {
                Color(
                    red: Double(brushSettings.color.x),
                    green: Double(brushSettings.color.y),
                    blue: Double(brushSettings.color.z),
                    opacity: Double(brushSettings.color.w)
                )
            },
            set: { newColor in
                // Convert SwiftUI Color back to SIMD4<Float>
                if let components = newColor.cgColor?.components, components.count >= 3 {
                    brushSettings.color = SIMD4<Float>(
                        Float(components[0]),
                        Float(components[1]),
                        Float(components[2]),
                        components.count >= 4 ? Float(components[3]) : 1.0
                    )
                }
            }
        )
    }

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "pencil.tip")
                    .font(.system(size: 14))
                    .foregroundColor(.white)
                Text("\(Int(brushSettings.size))pt")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.white)
                    .frame(width: 50, alignment: .leading)
            }

            Slider(
                value: $brushSettings.size,
                in: BrushSettings.minSize...BrushSettings.maxSize,
                step: 1.0
            )
            .tint(.white)

            // Color Picker
            ColorPicker("Stroke Color", selection: strokeColor, supportsOpacity: false)
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)

            // Culling Box Size Slider (Test)
            VStack(spacing: 4) {
                HStack(spacing: 8) {
                    Image(systemName: "viewfinder")
                        .font(.system(size: 14))
                        .foregroundColor(.white)
                    Text("Culling: \(String(format: "%.2fx", brushSettings.cullingMultiplier))")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.white)
                }
                Slider(
                    value: Binding(
                        get: { brushSettings.cullingMultiplier },
                        set: { newValue in
                            // Snap to 1.0, 0.5, or 0.25
                            if newValue >= 0.75 {
                                brushSettings.cullingMultiplier = 1.0
                            } else if newValue >= 0.375 {
                                brushSettings.cullingMultiplier = 0.5
                            } else {
                                brushSettings.cullingMultiplier = 0.25
                            }
                        }
                    ),
                    in: 0.25...1.0
                )
                .tint(.white)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
        .cornerRadius(20)
    }
}

#Preview {
    ContentView()
}
