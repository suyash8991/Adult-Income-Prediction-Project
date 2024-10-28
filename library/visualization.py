import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(table, challenge, strategy):
    """Plot confusion matrix for Random Forest model."""
    # if challenge == 'Class Imbalance':
    #     table = tables['imbalance']
    # elif challenge == 'Missing Values':
    #     table = tables['missing_values']
    # else:
    #     table = tables['outliers']
    
    strategy_index = {
        'No Change': 0,
        'SMOTE': 1,
        'Dropping Missing Data': 1,
        'Winsorizing': 1,
        'Cost Sensitive Learning': 2,
        'Imputing Data': 2,
        'Dropping Outlier': 2
    }.get(strategy, 0)
    
    print(f"\nChallenge {challenge} - Strategy {strategy}: Confusion Matrix for Random Forest\n")
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        table['Random Forest']['Confusion Matrix'][strategy_index], 
        annot=True, fmt='d', cmap='Blues', cbar=False
    )
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.show()