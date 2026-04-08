import matplotlib.pyplot as plt

# Your results (fill based on what YOU tested)
results = {
    "All Features": 0.12,
    "Behavior Only": 0.32,
    "Product + Income": 0.54,
    "Frequency + Income": 0.59,
    "Loyalty + Income": 0.60  
}

# Extract keys and values
features = list(results.keys())
scores = list(results.values())

# Plot
plt.figure(figsize=(10, 5))
colors = ['blue' if s < 0.6 else 'green' for s in scores]

plt.bar(features, scores, color=colors)

plt.xlabel("Feature Combinations")
plt.ylabel("Silhouette Score")
plt.title("KMeans Feature Comparison")

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("images/kmeans_feature_comparison.png", dpi=300)
plt.show()


segment_counts = df['Customer_Segment'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(segment_counts.index, segment_counts.values)

plt.xlabel("Customer Segments")
plt.ylabel("Number of Customers")
plt.title("Customer Segmentation Distribution")

plt.xticks(rotation=30)
plt.tight_layout()

# SAVE IMAGE
plt.savefig("outputs/customer_segments.png", dpi=300)

plt.show()