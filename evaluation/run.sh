###
# python 3.8.20, java

# cd /aifs4su/ziyaz/Audio-Flan-eval/evaluation
# bash run.sh

if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found!"
    exit 1
fi

echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "Installation complete."
