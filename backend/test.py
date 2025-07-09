# Add this before training
def check_label_distribution(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    fraud_count = sum(1 for seller in data if seller['labels']['is_fraud'] == 1)
    total = len(data)
    
    print(f"\nDataset: {json_file}")
    print(f"Total sellers: {total}")
    print(f"Fraud sellers: {fraud_count} ({fraud_count/total*100:.1f}%)")
    print(f"Legitimate sellers: {total-fraud_count} ({(total-fraud_count)/total*100:.1f}%)")
    
    # Check specific patterns
    patterns = ['is_network_member', 'has_price_coordination', 'has_inventory_sharing', 
                'has_registration_cluster', 'exit_scam_risk']
    
    for pattern in patterns:
        count = sum(1 for seller in data if seller['labels']['specific_patterns'][pattern] == 1)
        print(f"{pattern}: {count} ({count/total*100:.1f}%)")

# Check all your datasets
check_label_distribution('train_sellers.json')
check_label_distribution('val_sellers.json')
check_label_distribution('test_sellers.json')