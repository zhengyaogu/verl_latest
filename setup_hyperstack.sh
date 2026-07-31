set -e

git config --global user.email "zhengyao.gu30@gmail.com"
git config --global user.name "Zhengyao Gu"

huggingface-cli login --token "${HF_TOKEN:?set HF_TOKEN to your Hugging Face access token}"

echo "Done!"
