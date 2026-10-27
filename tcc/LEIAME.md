# Texto do TCC (LaTeX)

Fontes do TCC *Assessing the Impact of Data Leakage in Machine Learning
Pipelines for Regression Tasks*, na classe `infufrgs` (ECP, Trabalho de
Conclusão, em inglês com resumo e resumo expandido em português).

| Arquivo | Conteúdo |
|---|---|
| `main.tex` | Preâmbulo, metadados, abstract/resumo, lista de abreviaturas e ordem dos capítulos |
| `cap1-introducao.tex` | 1 Introduction |
| `cap2-fundamentacao.tex` | 2 Theoretical Background |
| `cap3-trabalhos-relacionados.tex` | 3 Related Works |
| `cap4-metodologia.tex` | 4 Methodology |
| `cap5-resultados.tex` | 5 Results and Discussion |
| `cap6-conclusao.tex` | 6 Conclusion |
| `apendice-a-tabelas.tex` | Apêndice A: tabelas e figuras detalhadas |
| `resumo-expandido.tex` | Apêndice B: resumo expandido em português |
| `tabelas/`, `figuras/` | Gerados por `scripts/gerar_tabelas_figuras.py` |
| `scripts/estatisticas.json` | Todos os números citados no texto (gerado pelo script) |

## Compilar

```bash
cd tcc
latexmk -pdf main.tex
```

No Overleaf: compacte a pasta `tcc/` (sem `scripts/`, se preferir) e use
**New Project → Upload Project**; compilador pdfLaTeX, documento principal
`main.tex`.

## Regenerar tabelas e figuras

Os números dos capítulos 4 e 5 vêm dos CSVs brutos em `results/`. Se os
experimentos forem refeitos, rode a partir da raiz do repositório:

```bash
pip install pandas scipy matplotlib
python tcc/scripts/gerar_tabelas_figuras.py
```

O texto cita alguns números diretamente (por exemplo, médias e p-valores);
confira-os contra `scripts/estatisticas.json` depois de regenerar.
