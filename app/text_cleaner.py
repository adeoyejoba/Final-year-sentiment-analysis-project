import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r"@\w+|\#", '', text)  # remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text
if __name__ == "__main__":
    test = "Hey there! Check this out: https://example.com #cool @user"
    print(clean_text(test))
