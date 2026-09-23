========================================================================
DOCUMENTAÇÃO DO PROJETO - DATA LEAKAGE ANALYSIS (PIPELINE DE REGRESSÃO)
========================================================================

VISÃO GERAL
------------------------------------------------------------------------
O projeto testa, de forma empírica, o impacto de vazamento de dados
(data leakage) em diferentes etapas de um pipeline de Machine Learning
de REGRESSÃO. Para cada etapa (normalização, imputação de valores
faltantes, seleção de features e ajuste de hiperparâmetros), o código
roda duas versões do mesmo experimento:

  - "leak"   -> a etapa é ajustada (fit) usando TODO o conjunto de
                dados (treino + teste), o que causa vazamento de
                informação do teste para o treino.
  - "noleak" -> a etapa é ajustada (fit) usando SOMENTE o conjunto de
                treino, e depois aplicada (transform) no teste, sem
                vazamento.

Os resultados de cada teste são salvos em um arquivo .csv, permitindo
comparar as métricas de regressão (MAE, MSE, RMSE, R²) obtidas com e
sem vazamento, para vários algoritmos e datasets do PMLB
(Penn Machine Learning Benchmarks).

Estrutura de arquivos:
  main.py              -> ponto de entrada, escolhe qual teste rodar
  model_utils.py        -> funções e configurações compartilhadas
  utils.py              -> funções matemáticas de baixo nível para
                            simular dados faltantes (MAR/MNAR)
  normalization.py      -> teste de vazamento na normalização (scaler)
  imputation.py         -> teste de vazamento na imputação de dados
  feature_selection.py  -> teste de vazamento na seleção de features
  tuning.py              -> teste de vazamento no ajuste de hiperparâmetros


========================================================================
1. main.py
========================================================================
Ponto de entrada do projeto. Não contém lógica de ML, apenas orquestra
qual teste será executado via linha de comando.

Conteúdo:
  - Importa as quatro funções principais de teste (uma de cada arquivo
    dentro do pacote "tests"): run_normalization_test,
    run_feature_selection_test, run_hyperparameter_tuning_test e
    run_value_imputation_test.
  - TESTS: dicionário que mapeia um nome de teste (string) para a
    função correspondente. É esse dicionário que permite escolher o
    teste pelo nome digitado no terminal.
  - run_tests(test_type):
      Recebe o nome do teste (string), busca a função em TESTS.
      Se o nome não existir, imprime um erro e a lista de testes
      disponíveis. Se existir, executa a função (test()).
  - Bloco "if __name__ == '__main__':"
      Lê o argumento passado na linha de comando (sys.argv[1]).
      Exige exatamente 1 argumento além do nome do script; caso
      contrário, mostra a forma correta de uso e encerra o programa
      (sys.exit(1)).
      Exemplo de uso no terminal:
          python main.py normalization
          python main.py feature_selection
          python main.py tuning
          python main.py imputation


========================================================================
2. model_utils.py
========================================================================
Arquivo central do projeto. Concentra as configurações globais
(algoritmos, datasets, número de repetições) e as funções utilitárias
usadas por todos os testes: criação de estimadores, definição das
métricas de avaliação, formatação dos resultados e simulação de
dados faltantes.

--- Configuração inicial ---
  signal_handler(sig, frame):
      Função chamada quando o usuário interrompe a execução com
      Ctrl+C (sinal SIGINT). Imprime "Exiting gracefully..." e encerra
      o programa de forma limpa (sys.exit(0)), em vez de deixar o
      Python mostrar um traceback de KeyboardInterrupt.
      A linha "signal.signal(signal.SIGINT, signal_handler)" registra
      essa função como o handler oficial do sinal assim que o módulo
      é importado.

--- Constantes globais ---
  target_algorithms:
      Lista com as siglas dos algoritmos de regressão usados em TODOS
      os testes: "LR" (regressão linear), "KNN" (k-vizinhos), "DT"
      (árvore de decisão), "RF" (random forest) e "SVR" (support
      vector regressor). "NN" (rede neural / MLP) está implementado
      em get_estimator, mas comentado nessa lista, ou seja, não é
      usado por padrão nos testes (pode ser reativado removendo o #).

  additional_15_ds:
      Lista de nomes de datasets do PMLB que serão usados nos testes.
      Atualmente só "560_bodyfat" está ativo; os demais estão
      comentados (#) e podem ser reativados removendo o #. Esses
      datasets têm alvo (target) contínuo, adequados para regressão.
      fetch_data(nome) (importado de "pmlb") baixa/carrega o dataset
      pelo nome.

  NUM_TRIALS = 6:
      Número de repetições de cada experimento (cada repetição usa
      uma seed diferente para o split/cv, permitindo calcular a
      variabilidade dos resultados).

--- Funções de criação de modelos e métricas ---

  get_estimator(alg):
      Recebe a sigla do algoritmo (string) e devolve uma instância do
      modelo de regressão do scikit-learn correspondente, já com
      hiperparâmetros padrão definidos:
        "LR"  -> LinearRegression()
        "KNN" -> KNeighborsRegressor()
        "DT"  -> DecisionTreeRegressor(random_state=42)
        "RF"  -> RandomForestRegressor(random_state=42)
        "SVR" -> SVR(kernel="linear", C=1)
        "NN"  -> MLPRegressor(hidden_layer_sizes=(100,),
                               random_state=42, max_iter=500)
      Se a sigla não for reconhecida, levanta ValueError.

  get_scoring():
      Devolve a lista de nomes de métricas (strings do scikit-learn)
      usadas em cross_validate/GridSearchCV:
        "neg_mean_absolute_error", "neg_mean_squared_error",
        "neg_root_mean_squared_error", "r2".
      Essas métricas são "neg_" (negativas) porque o scikit-learn
      sempre maximiza o score internamente; erros de regressão
      (que devem ser minimizados) são representados como negativos
      para caber nessa convenção.

  multiple_score(y_true, y_pred):
      Calcula manualmente 4 métricas de regressão a partir dos
      valores reais (y_true) e previstos (y_pred), devolvendo um
      dicionário com chaves "test_mae", "test_mse", "test_rmse" e
      "test_r2". Usada quando o modelo é treinado/avaliado
      manualmente (fora do cross_validate), como em
      feature_selection.py.

  make_dict(results):
      Recebe o atributo cv_results_ de um GridSearchCV (dicionário
      com um vetor de médias e um vetor de "ranks" por combinação de
      hiperparâmetros testada) e extrai apenas os valores da MELHOR
      combinação (rank == 1) para cada métrica, devolvendo um
      dicionário simples com "test_mae", "test_mse", "test_rmse" e
      "test_r2" (já convertidos para positivo, pois os valores
      "neg_*" do GridSearchCV são negativos). Usado em tuning.py para
      resumir o resultado do cenário "com vazamento" (o GridSearchCV
      roda direto nos dados completos).

  make_dict_mean(results):
      Recebe o dicionário devolvido por cross_validate (que contém um
      ARRAY de scores, um valor por fold de validação cruzada) e
      calcula a MÉDIA de cada métrica entre os folds, devolvendo um
      dicionário escalar com "test_mae", "test_mse", "test_rmse" e
      "test_r2" (também já convertidos de negativo para positivo nos
      casos "neg_*"). É essencial usar essa função sempre que o
      resultado vier de cross_validate, porque os valores brutos são
      arrays e não podem ser atribuídos diretamente a uma posição de
      um array numpy escalar (isso causa
      "ValueError: setting an array element with a sequence").
      Usada em normalization.py, imputation.py e no cenário
      "sem vazamento" de tuning.py.

--- Funções de simulação de dados faltantes ---

  produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
      Função central que gera valores faltantes artificiais em uma
      matriz de dados X, segundo um mecanismo de "missingness"
      escolhido:
        "MCAR"      -> Missing Completely At Random: cada valor tem a
                       mesma probabilidade (p_miss) de faltar,
                       independente de qualquer coisa (implementado
                       inline, sorteio uniforme direto).
        "MAR"       -> Missing At Random: delega para MAR_mask
                       (definida em utils.py); a chance de faltar
                       depende dos valores de OUTRAS variáveis
                       (que permanecem completas).
        "MNAR" com opt="logistic"   -> delega para MNAR_mask_logistic.
        "MNAR" com opt="quantile"   -> delega para MNAR_mask_quantiles.
        "MNAR" com opt="selfmasked" -> delega para
                                        MNAR_self_mask_logistic.
      Converte X para tensor PyTorch (se vier como numpy array),
      aplica a máscara de valores faltantes (substituindo por NaN) e
      devolve um dicionário com:
        'X_init'   -> dados originais completos (como tensor double)
        'X_incomp' -> dados com NaN inseridos
        'mask'     -> máscara booleana indicando onde os valores
                      foram removidos (True = valor faltante)

  get_imputer(imp):
      Recebe o nome de uma estratégia de imputação (string) e devolve
      o imputador do scikit-learn correspondente:
        "mean"      -> SimpleImputer(strategy='mean')
        "median"    -> SimpleImputer(strategy='median')
        "KNN"       -> KNNImputer(n_neighbors=5)
        "iterative" -> IterativeImputer(random_state=42)
        (qualquer outro valor cai no "else" e devolve
         SimpleImputer(strategy='mean') como padrão)

  insert_missing_in_df(df, perc):
      Recebe um DataFrame com uma coluna 'target' e uma porcentagem
      (perc). Separa o target, aplica produce_NA nas demais colunas
      usando o mecanismo "MAR" com p_obs=0.9 (90% das variáveis
      permanecem completas e são usadas para determinar onde os
      valores faltarão nas demais 10%), reconstrói um novo DataFrame
      com os valores faltantes inseridos e devolve a coluna 'target'
      original (sem NaN) junto. É essa função que o imputation.py usa
      para criar datasets com dados faltantes controlados antes de
      testar as estratégias de imputação.


========================================================================
3. utils.py
========================================================================
Biblioteca de apoio matemático (baseada em PyTorch) usada por
produce_NA em model_utils.py para simular diferentes mecanismos de
dados faltantes (MAR e MNAR) e, secundariamente, para calcular
métricas de erro de imputação. Não depende do restante do projeto e
poderia ser reaproveitada em outro contexto.

  nanmean(v, *args, **kwargs):
      Versão do np.nanmean para tensores PyTorch: substitui NaN por 0
      antes de somar e divide pela quantidade de valores que NÃO são
      NaN, obtendo a média ignorando os valores faltantes.

  quantile(X, q, dim=None):
      Calcula o quantil q (entre 0 e 1) de um tensor X usando
      kthvalue do PyTorch (equivalente ao quantile do numpy, mas
      compatível com tensores).

  pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
      Estima um valor de regularização (epsilon) a partir da mediana
      (por padrão) das distâncias euclidianas ao quadrado entre pares
      de linhas de X, usado como heurística em métodos de imputação
      baseados em Sinkhorn/OT (não usado diretamente pelos testes de
      regressão, mas mantido como utilitário). Para datasets grandes,
      amostra no máximo max_points linhas para evitar estourar
      memória.

  MAE(X, X_true, mask) / RMSE(X, X_true, mask):
      Calculam o erro absoluto médio / erro quadrático médio (raiz)
      APENAS nas posições marcadas como faltantes pela mask, comparando
      os valores imputados (X) com os valores verdadeiros originais
      (X_true). Funcionam tanto com tensores PyTorch quanto com
      arrays numpy (checam torch.is_tensor(mask) para decidir qual
      caminho seguir).

--- Mecanismos de dados faltantes ---

  MAR_mask(X, p, p_obs):
      Implementa o mecanismo "Missing At Random": escolhe
      aleatoriamente um subconjunto de variáveis que permanecerão
      totalmente observadas (proporção p_obs) e usa essas variáveis
      como entrada de um modelo logístico (com pesos aleatórios) para
      decidir a probabilidade de falta nas variáveis restantes.
      Os coeficientes do modelo logístico vêm de pick_coeffs, e o
      intercepto é ajustado por fit_intercepts para que a proporção
      final de valores faltantes seja próxima de p.

  MNAR_mask_logistic(X, p, p_params=.3, exclude_inputs=True):
      Implementa "Missing Not At Random" via modelo logístico. Se
      exclude_inputs=True, separa as variáveis em um grupo "de
      entrada" (usado como preditor do modelo logístico) e um grupo
      "mascarado" (que recebe valores faltantes segundo esse modelo);
      as variáveis de entrada também recebem uma máscara MCAR extra,
      fazendo com que a falta de um valor dependa de outros valores
      possivelmente também faltantes (daí o "not at random"). Se
      exclude_inputs=False, todas as variáveis participam tanto como
      entrada quanto como alvo da máscara.

  MNAR_self_mask_logistic(X, p):
      Variante MNAR em que a probabilidade de uma variável faltar
      depende dela mesma (self-masking): cada coluna tem seu próprio
      modelo logístico univariado.

  MNAR_mask_quantiles(X, p, q, p_params, cut='both', MCAR=False):
      Implementa MNAR por censura de quantil: seleciona um
      subconjunto de variáveis e marca como faltantes os valores que
      estão nos quantis extremos (inferior, superior ou ambos,
      conforme o parâmetro cut) dessas variáveis, com probabilidade p.
      Se MCAR=True, adiciona por cima uma máscara aleatória extra
      sobre todas as variáveis.

  pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
      Gera pesos aleatórios (coeficientes) para os modelos logísticos
      usados nos mecanismos MAR/MNAR, normalizados para que a
      combinação linear resultante (W^T x) tenha variância unitária
      (evita que a "escala" dos dados distorça as probabilidades).

  fit_intercepts(X, coeffs, p, self_mask=False):
      Ajusta o intercepto do modelo logístico (por busca de raiz,
      via optimize.bisect da scipy) de forma que a proporção média de
      valores marcados como faltantes seja igual ao p desejado.


========================================================================
4. normalization.py
========================================================================
Testa o impacto do vazamento de dados quando a normalização
(StandardScaler) é ajustada com o dataset inteiro em vez de apenas
com o treino.

  run_normalization_test():
      Para cada dataset em additional_15_ds e cada algoritmo em
      target_algorithms, repete NUM_TRIALS vezes (variando a seed do
      KFold):

        Cenário "noleak" (sem vazamento):
          Monta um Pipeline do scikit-learn com dois passos:
          StandardScaler seguido do estimador (regressor). Ao rodar
          cross_validate com esse pipeline, o scaler é ajustado
          (fit) DENTRO de cada fold de treino e só depois aplicado
          (transform) no fold de validação -- ou seja, o scaler nunca
          "vê" os dados de validação durante o ajuste.

        Cenário "leak" (com vazamento):
          Ajusta um StandardScaler UMA ÚNICA VEZ usando TODO o
          conjunto de dados (train_data completo, antes da divisão em
          folds) e transforma os dados inteiros. Só depois roda
          cross_validate com um Pipeline contendo apenas o
          estimador (sem scaler) sobre esses dados já normalizados
          com informação vazada de todos os folds.

      Em ambos os cenários, os scores de cada fold são resumidos pela
      make_dict_mean (média entre os 4 folds internos), e o resultado
      (dicionário com as 4 métricas + metadados: iteration, alg, leak,
      ds) é acumulado num DataFrame df_norm, salvo incrementalmente em
      "normalization_results.csv" a cada iteração (assim, mesmo que o
      script pare no meio, os resultados parciais não se perdem).


========================================================================
5. imputation.py
========================================================================
Testa o impacto do vazamento de dados quando a imputação de valores
faltantes é ajustada com o dataset inteiro em vez de apenas com o
treino, variando também o percentual de valores faltantes e a
estratégia de imputação usada.

  run_value_imputation_test():
      Loop aninhado: para cada dataset -> para cada percentual de
      valores faltantes em perc_miss ([0.05, 0.1, 0.2, 0.3]) -> gera
      um dataset com valores faltantes via insert_missing_in_df ->
      para cada algoritmo em target_algorithms -> para cada
      estratégia de imputação em imp_list (['mean', 'median', 'KNN'];
      'iterative' está disponível em get_imputer mas comentado aqui)
      -> repete NUM_TRIALS vezes:

        Cenário "noleak" (sem vazamento):
          Pipeline com três passos: imputer -> StandardScaler ->
          estimador. O cross_validate ajusta o imputer e o scaler
          dentro de cada fold de treino, sem tocar no fold de
          validação durante o fit. O resultado (array por fold) é
          resumido com make_dict_mean para virar um dicionário
          escalar.

        Cenário "leak" (com vazamento):
          Ajusta o imputer UMA ÚNICA VEZ com TODO o conjunto de
          dados (fit + transform em train_data inteiro) antes de
          entrar no cross_validate. Depois monta um Pipeline só com
          StandardScaler + estimador sobre esses dados já imputados
          "vazados", e roda cross_validate normalmente. Também
          resumido com make_dict_mean.

      Os dois dicionários de resultado recebem metadados (ds,
      iteration, alg, leak, imputer, perc-miss) e são concatenados em
      df_impute, salvo incrementalmente em "imputation_results.csv".


========================================================================
6. feature_selection.py
========================================================================
Testa o impacto do vazamento de dados quando a seleção de features
(SelectPercentile com o teste estatístico f_regression) é ajustada com
o dataset inteiro em vez de apenas com o treino.

  run_feature_selection_test():
      Para cada dataset e cada algoritmo em target_algorithms, e para
      cada percentual de features a manter em percentile_list
      ([1, 5, 10, 20]):

        - Calcula actual_perc: se o percentual solicitado resultaria
          em menos de 1 feature selecionada (dataset com poucas
          colunas), ajusta automaticamente o percentual mínimo
          necessário para manter ao menos 1 feature.

        - Cria leak_feature_selector (SelectPercentile com
          f_regression) e o ajusta (fit_transform) com TODO o X e y
          do dataset -- esse é o seletor que será reutilizado em
          todas as repetições do cenário "leak", carregando
          informação do dataset inteiro (incluindo o que depois vira
          "teste").

        - Se a seleção resultar em ao menos 1 feature (checagem
          len(X_leak[0]) > 0), repete NUM_TRIALS vezes:

            Faz um train_test_split (80/20) do dataset original.

            Cenário "noleak": cria um NOVO seletor de features
            (no_leak_feature_selector), ajustado (fit_transform)
            SOMENTE com os dados de treino da repetição atual; o
            conjunto de teste é apenas transformado (transform),
            nunca usado para decidir quais features são relevantes.
            Treina o estimador nos dados de treino já filtrados e
            avalia no teste, obtendo as métricas via multiple_score.

            Cenário "leak": reaproveita o leak_feature_selector
            (ajustado no início, com o dataset inteiro) para
            transformar tanto o treino quanto o teste da repetição
            atual -- ou seja, o teste está sendo transformado por um
            seletor que já "viu" os próprios dados de teste durante o
            ajuste original. Treina e avalia da mesma forma, também
            via multiple_score.

      Diferente de normalization.py e imputation.py, aqui as métricas
      NÃO vêm de cross_validate, e sim calculadas manualmente por
      multiple_score(y_true, y_pred) sobre a previsão no conjunto de
      teste do split -- por isso não é necessário usar make_dict_mean
      (o resultado já é escalar).

      Os resultados de cada repetição (dicionários com as métricas +
      metadados: ds, iteration, alg, leak, perc) são concatenados em
      df_feat, salvo incrementalmente em
      "feature_selection_results.csv".


========================================================================
7. tuning.py
========================================================================
Testa o impacto do vazamento de dados quando a busca de
hiperparâmetros (GridSearchCV) é feita de forma "não aninhada" (o
próprio conjunto usado para escolher os melhores hiperparâmetros
também é usado para reportar a performance final) versus "aninhada"
(nested cross-validation, onde a escolha de hiperparâmetros e a
avaliação final usam dados diferentes).

  run_hyperparameter_tuning_test():
      hyperparameterDict:
          Dicionário que define, para cada sigla de algoritmo, a
          grade de hiperparâmetros que o GridSearchCV vai testar:
            "KNN" -> n_neighbors e metric (distância)
            "SVR" -> C (regularização) e kernel
            "LR"  -> fit_intercept e positive (parâmetros válidos de
                     LinearRegression; note que "LR" aqui é regressão
                     linear simples, sem regularização, por isso a
                     grade é mais enxuta que a dos outros algoritmos)
            "DT"  -> max_depth
            "RF"  -> n_estimators e max_depth
            "NN"  -> hidden_layer_sizes, activation, solver, alpha,
                     learning_rate, max_iter, learning_rate_init
                     (grade grande, usada apenas se "NN" for
                     descomentado em target_algorithms)
          As chaves desse dicionário precisam bater exatamente com as
          siglas usadas em target_algorithms (model_utils.py), senão
          "p_grid = hyperparameterDict[alg]" gera KeyError.

      Para cada dataset e cada algoritmo, repete NUM_TRIALS vezes:

        Cenário "leak" (com vazamento / não aninhado):
          Cria um GridSearchCV com outer_cv (4 folds) e roda
          clf.fit(train_data, train_target) usando TODO o dataset do
          loop externo. O GridSearchCV internamente já faz
          cross-validation para escolher os melhores hiperparâmetros,
          mas o "veredito final" (best_score_ / cv_results_) vem dos
          MESMOS dados usados na busca -- não há um conjunto
          verdadeiramente de fora para validar a escolha. O resultado
          é resumido por make_dict(clf.cv_results_), que pega, para
          cada métrica, o valor médio da MELHOR combinação de
          hiperparâmetros encontrada (rank 1).

        Cenário "noleak" (sem vazamento / aninhado -- nested CV):
          Cria um segundo GridSearchCV (clf) usando inner_cv (4
          folds) para a busca de hiperparâmetros, mas em vez de dar
          fit diretamente, passa esse GridSearchCV inteiro como
          "estimador" para cross_validate, usando outer_cv (outros 4
          folds) como divisão EXTERNA. Isso é a validação cruzada
          aninhada: em cada fold externo, o GridSearchCV é reajustado
          do zero (buscando hiperparâmetros só com o treino daquele
          fold externo) e avaliado no fold de teste externo, que
          nunca participa da escolha de hiperparâmetros. O resultado
          (um array por fold) é resumido com make_dict_mean.

      Assim como nos demais testes, os resultados recebem metadados
      (ds, iteration, alg, leak) e são concatenados em df_hyp, salvo
      incrementalmente em "hyperparameter_tuning_results.csv".

      Detalhe técnico importante: os dois GridSearchCV recebem
      scoring=scoring (lista das 4 métricas de regressão) e
      refit='neg_mean_squared_error' -- ou seja, embora todas as
      métricas sejam calculadas, o "melhor" modelo/hiperparâmetro é
      sempre escolhido com base no menor erro quadrático médio. Como
      scoring é uma lista (multi-métrica), refit PRECISA ser uma
      string indicando qual métrica usar para decidir o reajuste
      final; se scoring não for passado, refit como string causa
      erro (esse foi um dos bugs corrigidos no projeto).


========================================================================
FLUXO RESUMIDO DE CADA TESTE
========================================================================
  1. Escolher dataset (PMLB) e algoritmo de regressão.
  2. Repetir NUM_TRIALS vezes, variando a semente aleatória:
       a) Rodar a etapa (scaler/imputer/seletor de features/busca de
          hiperparâmetros) SEM vazamento -- ajustada apenas com dados
          de treino.
       b) Rodar a MESMA etapa COM vazamento -- ajustada com o
          dataset inteiro (incluindo o que seria "teste").
       c) Calcular métricas de regressão (MAE, MSE, RMSE, R²) para os
          dois cenários.
  3. Salvar todos os resultados em um .csv, incluindo metadados de
     qual dataset, algoritmo, repetição e cenário (leak=True/False)
     geraram cada linha.
  4. Comparar os .csv gerados (fora do escopo do código) para medir o
     quanto o vazamento infla artificialmente a performance reportada
     em cada etapa do pipeline.
========================================================================
