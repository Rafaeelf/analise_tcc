import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm 
import os
import sys
import io
import gc
import scikit_posthocs as sp 

# --- Configurações ---
CAMINHO_DIRETORIO_BRUTO = r'C:\Users\User\Downloads\ProjetoAnalise'
NOMES_ARQUIVOS_BRUTOS = {
    'bcel': 'bcel-data.csv', #Projeto BCEL
    'csv': 'csv-data.csv', #Projeto CSV
    'text': 'text-data.csv', #Projeto TEXT
}
DIRETORIO_SAIDA_GRAFICOS = r'C:\Users\User\Downloads\ProjetoAnalise\graficos_analise_completa'
os.makedirs(DIRETORIO_SAIDA_GRAFICOS, exist_ok=True)

COLUNA_IDENTIFICADORA_METODO = 'method_name'

# Parâmetros de amostragem para o Teste de Wilcoxon
PERCENTUAL_AMOSTRAGEM_COMPARACOES = 0.05 # Amostrar 5% dos dados de CADA commit para a comparação
MAX_AMOSTRAS_COMPARACOES = 500000        # Limite máximo de amostras por commit para a comparação

# Função principal para processar e analisar cada projeto
def analisar_projeto(nome_projeto, nome_arquivo_bruto):
    output_filename = os.path.join(DIRETORIO_SAIDA_GRAFICOS, f'analise_completa_{nome_projeto}.txt')
    original_stdout = sys.stdout
    sys.stdout = open(output_filename, 'w', encoding='utf-8')

    print(f"\n--- Processando Projeto: {nome_projeto.upper()} ---")

    caminho_completo_arquivo = os.path.join(CAMINHO_DIRETORIO_BRUTO, nome_arquivo_bruto)

    if not os.path.exists(caminho_completo_arquivo):
        print(f"ERRO: Arquivo '{caminho_completo_arquivo}' não encontrado. Pulando projeto.")
        sys.stdout.close()
        sys.stdout = original_stdout
        return

    try:
        #Carregar os dados brutos
        df_bruto = pd.read_csv(
            caminho_completo_arquivo,
            usecols=['committer_date', 'commit_hash', 'method_started_at', 'method_ended_at', COLUNA_IDENTIFICADORA_METODO],
            dtype={
                'committer_date': str,
                'commit_hash': str,
                COLUNA_IDENTIFICADORA_METODO: str
            }
        )
        print(f"Carregado {len(df_bruto)} linhas do {caminho_completo_arquivo}")

        #Preparar os dados
        df_bruto['committer_date'] = pd.to_datetime(df_bruto['committer_date'], errors='coerce', utc=True)
        df_bruto['method_started_at'] = pd.to_datetime(df_bruto['method_started_at'], errors='coerce', utc=True)
        df_bruto['method_ended_at'] = pd.to_datetime(df_bruto['method_ended_at'], errors='coerce', utc=True)
        df_bruto = df_bruto.dropna(subset=['committer_date', 'method_started_at', 'method_ended_at', COLUNA_IDENTIFICADORA_METODO])
        df_bruto['own_duration_timedelta'] = df_bruto['method_ended_at'] - df_bruto['method_started_at']
        df_bruto['own_duration_seconds'] = df_bruto['own_duration_timedelta'].dt.total_seconds()
        df_bruto = df_bruto[df_bruto['own_duration_seconds'] >= 0]
        df_bruto = df_bruto.dropna(subset=['own_duration_seconds'])

        commits_info = df_bruto[['commit_hash', 'committer_date']].drop_duplicates().sort_values(by='committer_date')
        commits_unicos_ordenados = commits_info['commit_hash'].tolist()
        commit_hash_to_date = commits_info.set_index('commit_hash')['committer_date'].dt.strftime('%Y-%m-%d').to_dict()

        print(f"Número de commits únicos encontrados: {len(commits_unicos_ordenados)}")

        #Análise Descritiva por Commit (incluindo percentis)
        print(f"\n--- Análise Descritiva da Duração por Commit para {nome_projeto.upper()} ---")

        analise_descritiva = df_bruto.groupby('commit_hash')['own_duration_seconds'].agg(
            media='mean',
            mediana='median',
            desvio_padrao='std',
            maior='max',
            menor='min',
            p90=lambda x: x.quantile(0.90),
            p95=lambda x: x.quantile(0.95),
            p99=lambda x: x.quantile(0.99)
        ).reset_index()

        analise_descritiva['committer_date'] = analise_descritiva['commit_hash'].map(commit_hash_to_date)
        analise_descritiva.rename(columns={'commit_hash': 'Commit Hash'}, inplace=True)
        analise_descritiva = analise_descritiva[['Commit Hash', 'committer_date', 'media', 'mediana', 'desvio_padrao', 'maior', 'menor', 'p90', 'p95', 'p99']]

        analise_descritiva = analise_descritiva.round({
            'media': 6,
            'mediana': 6,
            'desvio_padrao': 6,
            'maior': 6,
            'menor': 6,
            'p90': 6,
            'p95': 6,
            'p99': 6
        })
        print(analise_descritiva.to_string(index=False))
        print("\n")

        #Gera grafico e Linha da Mediana
        if len(commits_unicos_ordenados) > 0:
            fig_width = max(10, len(commits_unicos_ordenados) * 0.4)
            plt.figure(figsize=(fig_width, 6))

            xtick_labels = [f"{hash[:7]}\n({commit_hash_to_date.get(hash, 'N/A')})" for hash in commits_unicos_ordenados]

            df_bruto['commit_hash_ordered'] = pd.Categorical(df_bruto['commit_hash'], categories=commits_unicos_ordenados, ordered=True)

            sns.stripplot(x='commit_hash_ordered', y='own_duration_seconds', data=df_bruto,
                          jitter=0.2, color='black', size=3, alpha=0.6)

            medianas_por_commit = df_bruto.groupby('commit_hash_ordered')['own_duration_seconds'].median().reindex(commits_unicos_ordenados)
            x_positions = range(len(commits_unicos_ordenados))

            plt.hlines(y=medianas_por_commit,
                               xmin=[pos - 0.3 for pos in x_positions],
                               xmax=[pos + 0.3 for pos in x_positions],
                               color='red', linewidth=2.5, zorder=5)

            plt.title(f'Distribuição da Duração dos Métodos por Commit - Projeto {nome_projeto.upper()}')
            plt.xlabel('Commit Hash (Data)')
            plt.ylabel('Duração (segundos)')
            plt.xticks(ticks=x_positions, labels=xtick_labels, rotation=45, ha='right', fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            nome_arquivo_scatter = os.path.join(DIRETORIO_SAIDA_GRAFICOS, f'grafico_com_mediana_desempenho_por_commit_{nome_projeto}.png')
            plt.savefig(nome_arquivo_scatter)
            print(f"Grafico plot com mediana por commit salvo como: {nome_arquivo_scatter}")
            plt.close()
        else:
            print(f"Não há dados suficientes para gerar o grafico para o projeto {nome_projeto}.")

        #Realizar Teste de Kruskal-Wallis
        if len(commits_unicos_ordenados) >= 3:
            grupos_duracao_por_commit = [df_bruto[df_bruto['commit_hash'] == commit]['own_duration_seconds'].dropna() for commit in commits_unicos_ordenados]
            grupos_duracao_por_commit = [g for g in grupos_duracao_por_commit if not g.empty]

            if len(grupos_duracao_por_commit) >= 3:
                stat_kw, p_value_kw = stats.kruskal(*grupos_duracao_por_commit)

                print(f"\nResultados do Teste de Kruskal-Wallis para {nome_projeto.upper()} (por Commit):")
                print(f"Estatística H: {stat_kw:.4f}")
                print(f"Valor p: {p_value_kw:.4f}")

                alfa = 0.05
                if p_value_kw < alfa:
                    print(f"Com p-value < {alfa}, rejeitamos a hipótese nula.")
                    print("Há uma diferença estatisticamente significativa na mediana da duração entre pelo menos dois dos commits.")

                    #Análise Post-Hoc de Dunn
                    print(f"\n--- Realizando Análise Post-Hoc de Dunn (com correção de Holm-Bonferroni) ---")
                    
                    df_kw_input = df_bruto[['commit_hash', 'own_duration_seconds']].copy()
                    
                    df_kw_input['commit_hash_category'] = pd.Categorical(df_kw_input['commit_hash'],
                                                                         categories=commits_unicos_ordenados,
                                                                         ordered=True)

                    dunn_results = sp.posthoc_dunn(df_kw_input, 
                                                   val_col='own_duration_seconds', 
                                                   group_col='commit_hash_category', 
                                                   p_adjust='holm')

                    short_hashes_dates = [f"{h[:7]} ({commit_hash_to_date.get(h, 'N/A')})" for h in commits_unicos_ordenados]
                    dunn_results.columns = short_hashes_dates
                    dunn_results.index = short_hashes_dates

                    print("\nTabela de P-valores do Teste de Dunn (corrigidos por Holm-Bonferroni):")
                    print(dunn_results.to_string())

                    print(f"\nComparações Pós-Hoc Significativas (alfa ajustado: {alfa:.4f}):")
                    significantes_dunn = []
                    for r_idx, row_hash in enumerate(commits_unicos_ordenados):
                        for c_idx, col_hash in enumerate(commits_unicos_ordenados):
                            if r_idx < c_idx: # Evitar duplicatas e auto-comparações
                                p_val = dunn_results.loc[f"{row_hash[:7]} ({commit_hash_to_date.get(row_hash, 'N/A')})", f"{col_hash[:7]} ({commit_hash_to_date.get(col_hash, 'N/A')})"]
                                if p_val < alfa:
                                    data1_es = df_bruto[df_bruto['commit_hash'] == row_hash]['own_duration_seconds'].dropna()
                                    data2_es = df_bruto[df_bruto['commit_hash'] == col_hash]['own_duration_seconds'].dropna()
                                    
                                    if not data1_es.empty and not data2_es.empty:
                                        # Amostragem para o cálculo do Effect Size para Mann-Whitney U
                                        n_samples_es1 = min(len(data1_es), MAX_AMOSTRAS_COMPARACOES)
                                        n_samples_es2 = min(len(data2_es), MAX_AMOSTRAS_COMPARACOES)
                                        
                                        sample_es1 = data1_es.sample(n=n_samples_es1, random_state=42)
                                        sample_es2 = data2_es.sample(n=n_samples_es2, random_state=42)

                                        try:
                                            # Usando Mann-Whitney U para calcular o Z
                                            stat_mw, p_mw = stats.mannwhitneyu(sample_es1, sample_es2, alternative='two-sided', method='auto')
                                            
                                            p_mw_capped = max(1e-300, min(1 - 1e-300, p_mw / 2)) 
                                            z_score_es = norm.ppf(p_mw_capped)
                                            
                                            if sample_es1.median() < sample_es2.median():
                                                z_score_es = abs(z_score_es) 
                                            else:
                                                z_score_es = -abs(z_score_es) 
                                            
                                            if np.isinf(z_score_es):
                                                effect_size_r = 'Inf'
                                            else:
                                                effect_size_r = z_score_es / np.sqrt(len(sample_es1) + len(sample_es2))
                                                effect_size_r = round(effect_size_r, 4)
                                                
                                        except ValueError as e_es:
                                            effect_size_r = f"Erro ES: {e_es}"
                                            
                                        significantes_dunn.append(f"   {row_hash[:7]} vs {col_hash[:7]}: p-valor = {p_val:.4f}, Effect Size r = {effect_size_r}")
                                    else:
                                        significantes_dunn.append(f"   {row_hash[:7]} vs {col_hash[:7]}: p-valor = {p_val:.4f}, Effect Size r = N/A (dados amostrados insuficientes)")

                    if significantes_dunn:
                        for s in significantes_dunn:
                            print(s)
                    else:
                        print("   Nenhuma diferença estatisticamente significativa encontrada após o teste de Dunn.")

                else:
                    print(f"Com p-value >= {alfa}, não há evidências suficientes para rejeitar a hipótese nula.")
                    print("Não há diferença estatisticamente significativa na mediana da duração entre os commits.")
            else:
                print(f"Menos de 3 grupos de dados válidos após filtragem para o projeto {nome_projeto}. Teste de Kruskal-Wallis não aplicável.")
        else:
            print(f"Menos de 3 commits únicos para o projeto {nome_projeto}. Teste de Kruskal-Wallis não aplicável.")

        #Teste de Wilcoxon Signed-Rank e Análise de Mudanças
        print(f"\n--- Resultados do Teste de Wilcoxon Signed-Rank e Análise de Mudanças para {nome_projeto.upper()} ---")

        if len(commits_unicos_ordenados) < 2:
            print(f"Aviso: Menos de 2 commits únicos para o projeto {nome_projeto}. Não é possível realizar Wilcoxon ou comparações de mudança.")
        else:
            wilcoxon_results = []
            alfa_wilcoxon = 0.05 / (len(commits_unicos_ordenados) - 1) # Correção de Bonferroni

            for i in range(1, len(commits_unicos_ordenados)):
                commit_anterior = commits_unicos_ordenados[i-1]
                commit_atual = commits_unicos_ordenados[i]

                df_anterior_full = df_bruto[df_bruto['commit_hash'] == commit_anterior].copy()
                df_atual_full = df_bruto[df_bruto['commit_hash'] == commit_atual].copy()

                # Estatísticas Resumidas
                print(f"\n--- Comparação Detalhada: {commit_anterior[:7]} ({commit_hash_to_date.get(commit_anterior, 'N/A')}) vs {commit_atual[:7]} ({commit_hash_to_date.get(commit_atual, 'N/A')}) ---")
                
                if not df_anterior_full.empty and not df_atual_full.empty:
                    mediana_anterior = df_anterior_full['own_duration_seconds'].median()
                    mediana_atual = df_atual_full['own_duration_seconds'].median()
                    media_anterior = df_anterior_full['own_duration_seconds'].mean()
                    media_atual = df_atual_full['own_duration_seconds'].mean()
                    p90_anterior = df_anterior_full['own_duration_seconds'].quantile(0.90)
                    p90_atual = df_atual_full['own_duration_seconds'].quantile(0.90)
                    p95_anterior = df_anterior_full['own_duration_seconds'].quantile(0.95)
                    p95_atual = df_atual_full['own_duration_seconds'].quantile(0.95)
                    p99_anterior = df_anterior_full['own_duration_seconds'].quantile(0.99)
                    p99_atual = df_atual_full['own_duration_seconds'].quantile(0.99)

                    def calcular_mudanca_percentual(valor_anterior, valor_atual):
                        if pd.isna(valor_anterior) or valor_anterior == 0:
                            return float('inf') if valor_atual > 0 else (0.0 if valor_atual == 0 else -float('inf'))
                        return ((valor_atual - valor_anterior) / valor_anterior) * 100

                    print(f"   Mediana (Anterior): {mediana_anterior:.6f} s")
                    print(f"   Mediana (Atual):     {mediana_atual:.6f} s")
                    print(f"   Mudança Mediana:     {calcular_mudanca_percentual(mediana_anterior, mediana_atual):.2f}%")
                    print(f"   Média (Anterior):    {media_anterior:.6f} s")
                    print(f"   Média (Atual):       {media_atual:.6f} s")
                    print(f"   Mudança Média:       {calcular_mudanca_percentual(media_anterior, media_atual):.2f}%")
                    print(f"   P90 (Anterior):      {p90_anterior:.6f} s")
                    print(f"   P90 (Atual):         {p90_atual:.6f} s")
                    print(f"   Mudança P90:         {calcular_mudanca_percentual(p90_anterior, p90_atual):.2f}%")
                    print(f"   P95 (Anterior):      {p95_anterior:.6f} s")
                    print(f"   P95 (Atual):         {p95_atual:.6f} s")
                    print(f"   Mudança P95:         {calcular_mudanca_percentual(p95_anterior, p95_atual):.2f}%")
                    print(f"   P99 (Anterior):      {p99_anterior:.6f} s")
                    print(f"   P99 (Atual):         {p99_atual:.6f} s")
                    print(f"   Mudança P99:         {calcular_mudanca_percentual(p99_anterior, p99_atual):.2f}%")
                else:
                    print("Dados insuficientes nos full dataframes para calcular estatísticas gerais.")

                
                # Agrupa por método e calcula a mediana da duração para cada método em cada commit
                agg_anterior = df_anterior_full.groupby(COLUNA_IDENTIFICADORA_METODO)['own_duration_seconds'].median().reset_index()
                agg_atual = df_atual_full.groupby(COLUNA_IDENTIFICADORA_METODO)['own_duration_seconds'].median().reset_index()

                # DataFrames agregados
                df_comparacao_paired = pd.merge(
                    agg_anterior,
                    agg_atual,
                    on=COLUNA_IDENTIFICADORA_METODO,
                    suffixes=('_anterior', '_atual'),
                    how='inner' # Queremos apenas métodos presentes em ambos os commits
                )
                
                n_pares_comuns = len(df_comparacao_paired)
                print(f"Pares de métodos comuns (medianas) para Wilcoxon e Top N: {n_pares_comuns}")

                if n_pares_comuns > 1:
                    data_anterior_wilcoxon = df_comparacao_paired['own_duration_seconds_anterior'].values
                    data_atual_wilcoxon = df_comparacao_paired['own_duration_seconds_atual'].values

                    try:
                        stat_w, p_value_w = stats.wilcoxon(data_anterior_wilcoxon, data_atual_wilcoxon, alternative='two-sided', method='auto')

                        p_val_capped_for_z = max(1e-300, min(1 - 1e-300, p_value_w / 2))
                        
                        z_score_wilcoxon = norm.ppf(p_val_capped_for_z)
                        
                        if np.median(data_anterior_wilcoxon) > np.median(data_atual_wilcoxon):
                                z_score_wilcoxon = -abs(z_score_wilcoxon)
                        else:
                                z_score_wilcoxon = abs(z_score_wilcoxon)

                        if np.isinf(z_score_wilcoxon):
                            effect_size_r_wilcoxon = 'Inf'
                        elif n_pares_comuns > 0:
                            effect_size_r_wilcoxon = z_score_wilcoxon / np.sqrt(n_pares_comuns)
                            effect_size_r_wilcoxon = round(effect_size_r_wilcoxon, 4)
                        else:
                            effect_size_r_wilcoxon = 'N/A (N=0)'

                        wilcoxon_results.append({
                            'Comparacao': f'{commit_anterior[:7]} vs {commit_atual[:7]}',
                            'N_Pares_Comuns': n_pares_comuns,
                            'Estatistica_W': round(stat_w, 4),
                            'Valor_p': round(p_value_w, 4),
                            'Significante': 'Sim' if p_value_w < alfa_wilcoxon else 'Não',
                            'Effect_Size_r': effect_size_r_wilcoxon
                        })
                    except ValueError as ve:
                        if "all zero differences" in str(ve):
                                wilcoxon_results.append({
                                    'Comparacao': f'{commit_anterior[:7]} vs {commit_atual[:7]}',
                                    'N_Pares_Comuns': n_pares_comuns,
                                    'Estatistica_W': 'N/A',
                                    'Valor_p': '1.0 (Todas as diferenças zero)',
                                    'Significante': 'Não',
                                    'Effect_Size_r': '0.0'
                                })
                        else:
                            wilcoxon_results.append({
                                    'Comparacao': f'{commit_anterior[:7]} vs {commit_atual[:7]}',
                                    'N_Pares_Comuns': n_pares_comuns,
                                    'Estatistica_W': 'Erro',
                                    'Valor_p': str(ve),
                                    'Significante': 'N/A',
                                    'Effect_Size_r': 'N/A'
                                })
                else:
                    wilcoxon_results.append({
                        'Comparacao': f'{commit_anterior[:7]} vs {commit_atual[:7]}',
                        'N_Pares_Comuns': n_pares_comuns,
                        'Estatistica_W': 'N/A',
                        'Valor_p': 'Dados insuficientes para Wilcoxon',
                        'Significante': 'N/A',
                        'Effect_Size_r': 'N/A'
                    })

                #Identificação dos métodos mais impactados
                if n_pares_comuns > 0:
                    df_comparacao_paired['duration_diff'] = df_comparacao_paired['own_duration_seconds_atual'] - df_comparacao_paired['own_duration_seconds_anterior']
                    
                    # Tratar divisão por zero para percent_change
                    df_comparacao_paired['percent_change'] = df_comparacao_paired.apply(
                        lambda row: (row['duration_diff'] / row['own_duration_seconds_anterior']) * 100 
                        if row['own_duration_seconds_anterior'] > 0 
                        else (np.inf if row['duration_diff'] > 0 else 0), axis=1
                    )
                    
                    df_comparacao_paired = df_comparacao_paired.replace([np.inf, -np.inf], np.nan).dropna(subset=['percent_change'])

                    if not df_comparacao_paired.empty:
                        top_regressoes = df_comparacao_paired.sort_values(by='percent_change', ascending=False).head(10)
                        if not top_regressoes.empty:
                            print(f"\nTop 10 Regressões de Desempenho (medianas) para {commit_anterior[:7]} vs {commit_atual[:7]}:")
                            print(top_regressoes[[COLUNA_IDENTIFICADORA_METODO, 'own_duration_seconds_anterior', 'own_duration_seconds_atual', 'duration_diff', 'percent_change']].round(6).to_string(index=False))

                        top_melhorias = df_comparacao_paired.sort_values(by='percent_change', ascending=True).head(10)
                        if not top_melhorias.empty:
                            print(f"\nTop 10 Melhorias de Desempenho (medianas) para {commit_anterior[:7]} vs {commit_atual[:7]}:")
                            print(top_melhorias[[COLUNA_IDENTIFICADORA_METODO, 'own_duration_seconds_anterior', 'own_duration_seconds_atual', 'duration_diff', 'percent_change']].round(6).to_string(index=False))
                    else:
                        print("\nNão foi possível gerar Top 10 Regressões/Melhorias para esta comparação (dados insuficientes após agregação/limpeza).")
                else:
                    print("\nNenhum método comum (medianas) encontrado para análise de Top N para esta comparação.")

                del agg_anterior, agg_atual, df_comparacao_paired
                gc.collect()

            df_wilcoxon_results = pd.DataFrame(wilcoxon_results)
            print(f"\n--- Resumo do Teste de Wilcoxon Signed-Rank para {nome_projeto.upper()} ---")
            print(df_wilcoxon_results.to_string(index=False))
            print(f"\nObservação: Foi aplicada uma correção de Bonferroni para comparações múltiplas.")
            print(f"Nível de significância ajustado (alfa): {alfa_wilcoxon:.4f}")
            print(f"As comparações de Wilcoxon e a análise de Top N foram baseadas nas medianas dos tempos de execução dos métodos comuns entre os commits.")


    except pd.errors.EmptyDataError:
        print(f"ERRO: O arquivo '{caminho_completo_arquivo}' está vazio ou não contém dados válidos. Pulando este projeto.")
    except KeyError as ke:
        print(f"ERRO: Coluna '{ke}' não encontrada no arquivo '{caminho_completo_arquivo}'. Verifique se os nomes das colunas estão corretos ou se a coluna '{COLUNA_IDENTIFICADORA_METODO}' existe e está sendo carregada.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao analisar o projeto {nome_projeto}: {e}")
        # Adicione o traceback completo para depuração
        import traceback
        traceback.print_exc(file=sys.stdout)
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Análise para o projeto {nome_projeto.upper()} concluída. Resultados exportados para: {output_filename}")


#Executar a análise para cada projeto
if __name__ == "__main__":
    for nome, arquivo_bruto in NOMES_ARQUIVOS_BRUTOS.items():
        analisar_projeto(nome, arquivo_bruto)

    print("\nAnálise completa para todos os projetos.")