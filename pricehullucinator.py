import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

class PriceHallucinator:
    def __init__(self, sample_size=50):
        """Initializes and trains a universal model using global encyclopedic data."""
        self.vectorizer = TfidfVectorizer()
        self.model = LinearRegression()
        
        # Pull global training keywords from Wikipedia
        items, prices = self._fetch_universal_training_data(sample_size)
        X_train = self.vectorizer.fit_transform(items)
        self.model.fit(X_train, prices)

    def _fetch_universal_training_data(self, sample_size):
        """Queries Wikipedia's public API to get a vast dictionary of real-world objects."""
        # Queries random encyclopedic articles to build a universal vocabulary
        url = f"https://wikipedia.org{sample_size}&format=json"
        items, prices = [], []
        
        try:
            res = requests.get(url, headers={'User-Agent': 'PriceHallucinatorBot/1.0'}, timeout=5)
            if res.status_code == 200:
                pages = res.json().get("query", {}).get("random", [])
                for i, page in enumerate(pages):
                    title = page.get("title", "").strip()
                    if title and not any(x in title.lower() for x in ["list of", "talk:", "wikipedia:"]):
                        items.append(title)
                        
                        # Mathematical pricing formula based on text complexity
                        # More words/characters usually correlates to a premium or specific item
                        base_math_price = 5.00 + (len(title) * 1.50) + ((i % 5) * 12.50)
                        
                        # Add keyword-based multiplier logic for common premium niches
                        title_lower = title.lower()
                        if any(w in title_lower for w in ["lego", "set", "edition", "system", "device"]):
                            base_math_price *= 3.5
                        if any(w in title_lower for w in ["chips", "soda", "candy", "album"]):
                            base_math_price *= 0.4
                            
                        prices.append(round(base_math_price, 2))
        except Exception:
            pass
            
        # Robust catch-all fallback list representing a wide span of items if offline
        if not items:
            items = ["Doritos Chips", "Star Wars Lego", "Apple iPhone", "Soft Drink Bottle", "Leather Couch", "Harry Potter Book"]
            prices = [4.50, 49.99, 999.00, 2.50, 450.00, 19.99]
            
        return items, prices

    def predict(self, product_name):
        """Processes any arbitrary retail text to calculate a hallucinated price."""
        if not product_name:
            return "$0.00"
        
        # Step 1: Base regression prediction
        X_new = self.vectorizer.transform([product_name])
        prediction = self.model.predict(X_new)[0]
        
        # Step 2: Post-processing modifier to dynamically bump value based on user words
        p_lower = product_name.lower()
        multiplier = 1.0
        if "lego" in p_lower or "set" in p_lower:
            multiplier += 4.5
        if "console" in p_lower or "pro" in p_lower:
            multiplier += 10.0
        if "pack" in p_lower:
            multiplier += 1.5
        
        final_price = max(0.99, round(prediction * multiplier, 2))
        return f"${final_price:.2f}"
