//
//  ContentView.swift
//  Slate
//
//  Created by Penny Marshall on 10/27/25.
//

import SwiftUI

struct ContentView: View {
    // Reference to MetalView's coordinator for adding cards
    @State private var metalViewCoordinator: MetalView.Coordinator?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            MetalView(coordinator: $metalViewCoordinator)
            .edgesIgnoringSafeArea(.all)

            // Add Card Button
            Button(action: {
                print("🔘 Add Card button tapped")
                if metalViewCoordinator == nil {
                    print("❌ ERROR: Coordinator is nil!")
                } else {
                    print("✅ Coordinator exists, calling addCard()")
                    metalViewCoordinator?.addCard()
                }
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
            .padding(.top, 60)
            .padding(.trailing, 16)
        }
    }
}

#Preview {
    ContentView()
}
