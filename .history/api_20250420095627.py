from flask import Flask, request, jsonify
from check_complaint import check_complaint  # Import the complaint analysis function

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_complaint():
    data = request.json
    complaint = data.get('complaint')
    if not complaint:
        return jsonify({"error": "Complaint text is required"}), 400

    # Use the check_complaint function to analyze the complaint
    result = check_complaint(complaint)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)