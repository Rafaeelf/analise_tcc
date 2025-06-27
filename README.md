# Análise da Evolução do Desempenho de Software em Projetos Apache Commons

Este repositório contém o código e os recursos para a análise da evolução do desempenho em projetos de software de código aberto, focando em projetos da suíte Apache Commons (Text, CSV e BCEL). O objetivo é identificar e quantificar melhorias e degradações de desempenho ao longo do histórico de commits, utilizando uma abordagem estatística robusta.

## 🎯 Objetivos do Projeto

* **Quantificar e analisar** as variações de desempenho (melhorias e degradações) de métodos em projetos de software de código aberto ao longo de seus históricos de desenvolvimento, utilizando dados de duração de execução extraídos.
* **Correlacionar as alterações de desempenho identificadas** com as mudanças ocorridas nos commits, buscando insights sobre como refatorações, novas funcionalidades ou correções impactam a performance do software ao longo do tempo.
* **Aplicar e interpretar testes estatísticos não paramétricos** (como Kruskal-Wallis, Wilcoxon Signed-Rank e Dunn Post-Hoc) para detectar diferenças estatisticamente significativas nas distribuições de tempo de execução e avaliar o tamanho do efeito dessas mudanças em diferentes commits ou fases de desenvolvimento.

## ✨ Funcionalidades

O script Python automatiza um processo completo de análise de desempenho, incluindo:

* **Pré-processamento de Dados:** Leitura de arquivos CSV brutos contendo telemetria de execução de métodos, cálculo de durações e tratamento de dados faltantes.
* **Análise Descritiva por Commit:** Geração de estatísticas descritivas (média, mediana, percentis P90, P95, P99) da duração dos métodos para cada commit.
* **Visualização de Desempenho:** Criação de gráficos de dispersão (stripplots) que ilustram a distribuição da duração dos métodos por commit, com destaque para a linha da mediana.
* **Testes Estatísticos Robustos:**
    * **Kruskal-Wallis:** Para verificar diferenças globais na distribuição de desempenho entre todos os commits.
    * **Dunn Post-Hoc:** Para identificar quais pares específicos de commits diferem significativamente (com correção de Holm-Bonferroni).
    * **Wilcoxon Signed-Rank:** Para comparar as medianas de métodos comuns entre commits adjacentes e avaliar a significância de transições pontuais.
* **Cálculo de Tamanho do Efeito:** Quantificação da magnitude das diferenças de desempenho identificadas.
* **Identificação de Métodos Chave:** Listagem dos Top 10 métodos com as maiores regressões e melhorias percentuais entre commits adjacentes, destacando pontos críticos para otimização.
* **Geração de Logs Detalhados:** Toda a saída da análise (estatísticas, resultados de testes) é exportada para arquivos de texto para cada projeto.

## 🛠️ Tecnologias Utilizadas

O projeto é desenvolvido em Python, utilizando as seguintes bibliotecas:

* `pandas`: Manipulação e análise de dados tabulares.
* `numpy`: Operações numéricas de alta performance.
* `matplotlib`: Criação de gráficos estáticos.
* `seaborn`: Visualizações estatísticas atraentes.
* `scipy`: Funções para testes estatísticos (Kruskal-Wallis, Wilcoxon, distribuição normal).
* `scikit_posthocs`: Implementação de testes post-hoc (Dunn).
* `os`, `sys`, `io`, `gc`: Para interações com o sistema operacional, controle de I/O e gerenciamento de memória.

## ⚙️ Configuração e Execução

### Pré-requisitos

Certifique-se de ter o Python 3.x instalado.
Instale as bibliotecas necessárias usando pip:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-posthocs
