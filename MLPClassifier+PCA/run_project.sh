#!/bin/bash
# Este script funciona tanto em Windows (via Git Bash ou WSL) quanto em Unix/Linux

# Detectar o sistema operacional
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="Windows"
else
    OS="Unix"
fi

# Definir o nome do ambiente virtual
ENV_NAME="mnist_env"

# Obter o diretório do script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Definir o diretório raiz do projeto (um nível acima)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Função para executar comandos específicos do Windows
run_windows_commands() {
    python --version > /dev/null 2>&1 || { echo "Python não está instalado. Por favor, instale-o e tente novamente."; exit 1; }
    virtualenv --version > /dev/null 2>&1 || { echo "Virtualenv não está instalado. Instalando..."; pip install virtualenv; }
    [ ! -d "$ENV_NAME" ] && { echo "Criando ambiente virtual..."; virtualenv $ENV_NAME; } || echo "Ambiente virtual já existe."
    source $ENV_NAME/Scripts/activate
}

# Função para executar comandos específicos do Unix/Linux
run_unix_commands() {
    python3 --version > /dev/null 2>&1 || { echo "Python3 não está instalado. Por favor, instale-o e tente novamente."; exit 1; }
    virtualenv --version > /dev/null 2>&1 || { echo "Virtualenv não está instalado. Instalando..."; pip3 install virtualenv; }
    [ ! -d "$ENV_NAME" ] && { echo "Criando ambiente virtual..."; virtualenv $ENV_NAME; } || echo "Ambiente virtual já existe."
    source $ENV_NAME/bin/activate
}

# Executar comandos específicos do sistema operacional
[ "$OS" == "Windows" ] && run_windows_commands || run_unix_commands

# Função para verificar e instalar/atualizar dependências
check_and_install_dependencies() {
    echo "Verificando dependências..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    pip list
}

# Verificar e instalar/atualizar dependências
check_and_install_dependencies

# Adicionar o diretório raiz do projeto ao PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Mudar para o diretório do projeto
cd "$PROJECT_ROOT"

# Executar o script principal
echo "Executando o projeto..."
python -m MLPClassifier+PCA.main

# Desativar o ambiente virtual
deactivate

echo "Execução concluída. O ambiente virtual foi desativado."
