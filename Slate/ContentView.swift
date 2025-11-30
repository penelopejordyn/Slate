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
            .onChange(of: metalViewCoordinator) { newCoord in
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
