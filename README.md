# -dimport re
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = defaultdict(float)
        self.feature_probabilities = defaultdict(lambda: defaultdict(float))
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.classes = set()

    def train(self, documents):
        class_counts = defaultdict(int)
        for document, label in documents:
            self.classes.add(label)
            class_counts[label] += 1
            for word in self.extract_words(document):
                self.feature_counts[label][word] += 1
        total_documents = sum(class_counts.values())
        for label, count in class_counts.items():
            self.class_probabilities[label] = count / total_documents
            total_words = sum(self.feature_counts[label].values())
            for word, word_count in self.feature_counts[label].items():
                self.feature_probabilities[label][word] = word_count / total_words

    def predict(self, document):
        scores = {label: math.log(prob) for label, prob in self.class_probabilities.items()}
        for word in self.extract_words(document):
            for label in self.classes:
                scores[label] += math.log(self.feature_probabilities[label][word] + 1)
        return max(scores, key=scores.get)

    def extract_words(self, document):
        words = re.findall(r'\b\w+\b', document.lower())
        return set(words)

# 示例数据
training_data = [
    ("Python is a programming language", "programming"),
    ("Bananas are a fruit", "fruit"),
    ("Java is also a programming language", "programming"),
    ("Strawberries are a fruit", "fruit")
]

# 创建和训练分类器
classifier = NaiveBayesClassifier()
classifier.train(training_data)

# 测试分类器
test_documents = ["I like Python", "Apples are fruit"]
for doc in test_documents:
    print(f"\"{doc}\" is classified as \"{classifier.predict(doc)}\"")
