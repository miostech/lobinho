import json
import re

# Path to the file
file_path = 'Analise.rtf'

# Read the file content
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Regular expression to extract JSON blocks after "ZENVIA WEBHOOK"
json_pattern = re.compile(r'ZENVIA WEBHOOK (\\{.*?\\})')

# Extract all JSON-like blocks
json_blocks = json_pattern.findall(content)


# Define a function to validate the required fields
def validate_json_data(data):
    fields = {
        'status': data.get('messageStatus', {}).get('code'),
        'timestamp': data.get('timestamp'),
        'code': data.get('messageStatus', {}).get('code'),
        'reason': None,
        'to': data.get('message', {}).get('to'),
        'payload': json.dumps(data)
    }

    # Handling the nested causes[0]["reason"]
    causes = data.get('messageStatus', {}).get('causes', [])
    if causes and len(causes) > 0:
        fields['reason'] = causes[0].get('reason')

    # Identify missing fields
    missing_fields = [key for key, value in fields.items() if value is None]

    return fields, missing_fields


# Parse each JSON block and validate
results = []
for block in json_blocks:
    try:
        # Unescape the JSON block
        json_data = json.loads(block.replace('\\', ''))
        fields, missing = validate_json_data(json_data)
        results.append({
            'fields': fields,
            'missing': missing
        })
    except json.JSONDecodeError:
        # If it's not a valid JSON, skip it
        continue

# Print results
for result in results:
    print("Fields:", result['fields'])
    if result['missing']:
        print("Missing fields:", result['missing'])
    print("=" * 50)
