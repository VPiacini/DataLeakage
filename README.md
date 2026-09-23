# Data Leakage em Pipelines de Regressão

Avaliação empírica do impacto de **vazamento de dados** (*data leakage*) sobre a
estimativa de desempenho de modelos de regressão, em quatro etapas distintas de
um pipeline de Machine Learning: normalização, imputação de valores faltantes,
seleção de atributos e ajuste de hiperparâmetros.

Para cada etapa, o mesmo experimento roda em dois cenários:

| Cenário  | Como a etapa é ajustada |
|----------|-------------------------|
| `noleak` | `fit` **somente no treino**; o teste é apenas transformado (`transform`) |
| `leak`   | `fit` no **dataset inteiro** (treino + teste), contaminando a avaliação |

A diferença entre os dois cenários, medida em MAE, MSE, RMSE e R², quantifica o
quanto o vazamento infla artificialmente a performance reportada.

---

## Requisitos

- Python 3.10+
- Dependências fixadas em `requirements.txt` (scikit-learn 1.6.1, pmlb 1.0.1,
  pandas, numpy, scipy, torch, matplotlib, seaborn)

O `torch` é usado apenas por `utils.py`, na geração dos mecanismos de dados
faltantes MAR/MNAR. Não há treino em GPU em nenhum ponto do projeto.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Os datasets são baixados automaticamente do PMLB na primeira execução
(`pmlb.fetch_data`), portanto a primeira rodada exige conexão com a internet.

## Execução

Cada teste é disparado separadamente:

```bash
python main.py normalization
python main.py imputation
python main.py feature_selection
python main.py tuning
```

Os resultados são gravados de forma **incremental** em `results/<teste>_results.csv`
— uma linha por (dataset × algoritmo × repetição × cenário) e demais fatores do
teste. Interromper com `Ctrl+C` encerra de forma limpa e preserva o que já foi
calculado.

### Retomar uma execução interrompida

```bash
RESUME=1 python main.py tuning
```

Com `RESUME=1`, os blocos já presentes no CSV são pulados. É a forma recomendada
de rodar os testes mais caros (`tuning` e `imputation`) em várias sessões.

## Configuração

Todos os parâmetros são opcionais e definidos por variável de ambiente:

| Variável | Default | Descrição |
|---|---|---|
| `DATASETS` | os 20 da lista em `model_utils.py` | Lista separada por vírgula, ex.: `DATASETS=227_cpu_small,201_pol` |
| `ALGS` | `LR,KNN,DT,RF,SVR` | Lista separada por vírgula, ex.: `ALGS=SVR,KNN` |
| `NUM_TRIALS` | `6` | Repetições por configuração (seeds diferentes) |
| `N_JOBS` | `-1` | Paralelismo do CV / GridSearchCV (`-1` = todos os núcleos) |
| `SVR_IMPL` | `linear` | `linear` usa `LinearSVR` (rápido); `libsvm` usa o `SVR(kernel="linear")` original |
| `SVR_SCALE_Y` | `1` | Padroniza o alvo dentro do SVR; `0` desliga |
| `SVR_MAX_ROWS` | `5000` | Teto de linhas para SVR com kernel; `0` remove o teto |
| `SVR_CACHE_MB` | `1000` | Cache do libsvm, em MB |
| `RESUME` | `0` | `1` pula blocos já gravados no CSV |

Exemplo — rodar só o SVR em dois datasets, com 3 repetições:

```bash
DATASETS=1089_USCrime,505_tecator ALGS=SVR NUM_TRIALS=3 python main.py normalization
```

## Datasets

20 datasets de regressão do PMLB, listados em `model_utils.py` **ordenados por
número de instâncias** (47 a 22.784 linhas). A lista cobre desde datasets muito
pequenos, onde o efeito do vazamento tende a ser mais visível, até um caso de
alta dimensionalidade relativa (`505_tecator`, 240 linhas × 124 atributos).

Observações sobre a composição da lista:

- `529_pollen`, `225_puma8NH` e `1199_BNG_echoMonths` são **sintéticos** (gerados
  por simulação ou, no último caso, por um *Bayesian Network Generator*). Os
  demais são dados reais.
- `574_house_16H` foi removido por ser redundante: mesma base (censo dos EUA de
  1990) e mesmas 22.784 linhas de `218_house_8L`, apenas com um subconjunto de
  atributos escolhido artificialmente.
- `228_elevators` e `573_house_16L` não existem mais no PMLB 1.0; `fetch_data`
  falharia para eles.

## Os quatro experimentos

### `normalization`
`StandardScaler` ajustado no dataset inteiro (leak) contra ajustado por fold
dentro de um `Pipeline` (noleak). Métricas via `cross_validate`, média entre os
4 folds internos.

### `imputation`
Valores faltantes são inseridos artificialmente antes do experimento, variando:

- **Percentual faltante:** 5%, 10%, 20%, 30%
- **Imputadores:** média, mediana, KNN

No cenário leak o imputador é ajustado sobre a matriz completa; no noleak, dentro
do `Pipeline`, a cada fold. A simulação dos mecanismos MAR/MNAR está em
`utils.py`.

### `feature_selection`
`SelectPercentile(f_regression)` com percentis de **1%, 5%, 10% e 20%** dos
atributos. No leak, o seletor é ajustado uma vez sobre todo o dataset e reusado;
no noleak, reajustado a cada repetição apenas com o treino. Quando o percentil
resultaria em menos de um atributo, ele é elevado ao mínimo que garante 1 — e o
**mesmo** valor é usado nos dois cenários, para que a comparação seja justa.

Aqui as métricas são calculadas diretamente sobre a predição do conjunto de
teste (não via `cross_validate`).

### `tuning`
`GridSearchCV` **não aninhado** (leak: a mesma partição escolhe os
hiperparâmetros e reporta o desempenho) contra **validação cruzada aninhada**
(noleak: 4 folds internos para a busca, 4 externos para avaliar). Ambos usam
`refit='neg_mean_squared_error'`.

## Formato dos resultados

Colunas comuns aos quatro CSVs:

| Coluna | Significado |
|---|---|
| `test_mae`, `test_mse`, `test_rmse`, `test_r2` | Métricas no conjunto de teste |
| `ds` | Nome do dataset no PMLB |
| `alg` | `LR`, `KNN`, `DT`, `RF` ou `SVR` |
| `iteration` | Índice da repetição (0 … `NUM_TRIALS`-1) |
| `leak` | `True` = com vazamento, `False` = sem |

Colunas adicionais: `perc` (feature_selection); `imputer` e `perc-miss`
(imputation).

## Análise

```bash
python analyze_results.py                                   # todos os testes
python analyze_results.py --tests normalization imputation  # subconjunto
python analyze_results.py --input-dir results --output-dir plots
```

Gera, por teste, em `plots/<teste>/`:

- **Boxplots** da métrica por algoritmo, comparando os dois cenários
- **Gráficos de gap** (leak − noleak): o viés do vazamento isolado
- **Erro × tamanho do dataset**, com o eixo x em escala logarítmica, para
  verificar se o viés cresce em datasets menores
- **Tabelas resumo** em `plots/summaries/`

> MAE, MSE e RMSE são medidos na escala original do alvo e **não são comparáveis
> entre datasets** com alvos de magnitudes diferentes. Para a leitura entre
> datasets — em particular no gráfico de erro × tamanho — use o **R²**, que é
> adimensional.

O dicionário `DATASET_SIZES`, em `analyze_results.py`, guarda o `n_instances` de
cada dataset e precisa ser atualizado manualmente sempre que a lista em
`model_utils.py` mudar.

## Estrutura

```
main.py                 ponto de entrada; despacha o teste pedido na linha de comando
model_utils.py          configuração global, estimadores, métricas, I/O dos resultados
utils.py                mecanismos de dados faltantes MCAR/MAR/MNAR
normalization.py        experimento de normalização
imputation.py           experimento de imputação
feature_selection.py    experimento de seleção de atributos
tuning.py               experimento de ajuste de hiperparâmetros
analyze_results.py      leitura dos CSVs e geração de gráficos e tabelas
results/                CSVs gerados (ignorados pelo git)
plots/                  gráficos gerados (ignorados pelo git)
```

`results/` e `plots/` ficam fora do controle de versão: são inteiramente
reprodutíveis a partir do código e das instruções acima.

## Custo computacional

`normalization` e `feature_selection` rodam em tempo moderado na lista completa.
`tuning` e `imputation` são substancialmente mais caros — o gargalo é o SVR em
datasets grandes e o `KNNImputer` com percentuais altos de valores faltantes.
Para essas duas, use `RESUME=1` e, se necessário, fatie a execução com `DATASETS`
e `ALGS`.

## Créditos

A geração dos mecanismos de dados faltantes em `utils.py` é adaptada do código
que acompanha:

> MUZELLEC, B.; JOSSE, J.; BOYER, C.; CUTURI, M. Missing Data Imputation using
> Optimal Transport. *Proceedings of the 37th International Conference on Machine
> Learning (ICML)*, PMLR v. 119, p. 7130–7140, 2020.
> https://proceedings.mlr.press/v119/muzellec20a.html

Os datasets vêm do Penn Machine Learning Benchmarks:

> ROMANO, J. D.; LE, T. T.; LA CAVA, W. et al. PMLB v1.0: an open-source dataset
> collection for benchmarking machine learning methods. *Bioinformatics*, v. 38,
> n. 3, p. 878–880, 2022.
