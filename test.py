from presidio_analyzer import AnalyzerEngine

# Analyzer nur EINMAL initialisieren (teuer!)
analyzer = AnalyzerEngine()


def analyze_texts(texts: list[str], language: str = "en"):
    results_per_text = []
    for text in texts:
        results = analyzer.analyze(text=text, language=language)
        results_per_text.append({
            "text": text,
            "entities": [
                {"entity": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
                for r in results
            ]
        })
    return results_per_text


# Beispiel
texts = [
    "My name is Jonas and my email is jonas@example.com.",
    "My credit card is 1234-5678-9012-3456."
]

print(analyze_texts(texts))
