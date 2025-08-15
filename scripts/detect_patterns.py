import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cognition_log.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['CognitiveScore'] = df[['Mood', 'Focus', 'Memory']].mean(axis=1)

# Detect dips and spikes
df['Dip'] = df['CognitiveScore'].rolling(3).apply(lambda x: x[-1] < x.mean())
df['Spike'] = df['CognitiveScore'].rolling(3).apply(lambda x: x[-1] > x.mean() + 1)

# Plot
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['CognitiveScore'], label='Cognitive Score', marker='o')
plt.scatter(df[df['Dip'] == 1]['Date'], df[df['Dip'] == 1]['CognitiveScore'], color='red', label='Dip')
plt.scatter(df[df['Spike'] == 1]['Date'], df[df['Spike'] == 1]['CognitiveScore'], color='green', label='Spike')
plt.title("Cognitive Score with Detected Patterns")
plt.xlabel("Date")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("visuals/pattern_graph.png")
plt.show()
