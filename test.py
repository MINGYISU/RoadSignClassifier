import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def main():
    with open('history.pkl', 'rb') as f:
        history = pickle.load(f)

    sns.set(style="whitegrid")
    plt.rc('figure', autolayout=True)
    plt.rc('axes', labelweight='bold', labelsize='large',
            titleweight='bold', titlesize=18, titlepad=10)

    # show loss
    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


    # show accuracy
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    main()
