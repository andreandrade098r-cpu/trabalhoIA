"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SISTEMA HÍBRIDO FUZZY-NEURO-GENÉTICO PARA DIAGNÓSTICO DE DOENÇA CARDÍACA  ║
║   Dataset: Cleveland Heart Disease (UCI) — 303 pacientes reais               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pygad
import skfuzzy as fuzz
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, accuracy_score
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


# ── Caminhos resolvidos automaticamente ───────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "heart_cleveland.csv"
OUT_DIR = BASE_DIR / "resultados"
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# PASSO 1: CARREGAMENTO DO DATASET
# ═══════════════════════════════════════════════════════════════════════════

def carregar_dataset():
    colunas = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    df = pd.read_csv(CSV_PATH, header=None, names=colunas, na_values=["?", ""])
    df["ca"] = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].median())
    df["target"] = (df["num"] > 0).astype(int)
    print(
        f"  Dataset: {len(df)} pacientes | "
        f"Saudavel: {(df.target == 0).sum()} | "
        f"Doenca: {(df.target == 1).sum()} "
        f"({100 * df.target.mean():.1f}%)"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PASSO 2: PRE-PROCESSAMENTO
# ═══════════════════════════════════════════════════════════════════════════

def preparar_dados(df):
    feats = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    X = df[feats].values.astype(float)
    y = df["target"].values

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )

    scaler = MinMaxScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    print(
        f"  Treino: {len(y_tr)} | Validacao: {len(y_val)} | Teste: {len(y_te)}")
    return X_tr_s, X_val_s, X_te_s, y_tr, y_val, y_te, feats


# ═══════════════════════════════════════════════════════════════════════════
# PASSO 3: REDE NEURAL
# ═══════════════════════════════════════════════════════════════════════════

def treinar_rede_neural(X_train, y_train):
    mlp = MLPClassifier(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=3000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    return mlp


# ═══════════════════════════════════════════════════════════════════════════
# PASSO 4: FUZZY MAMDANI
# ═══════════════════════════════════════════════════════════════════════════

def _pertinencias(x, g):
    u = np.linspace(0, 1, 101)
    baixo = fuzz.trimf(u, [0.0, 0.0, g[1]])
    medio = fuzz.trimf(u, [g[0], g[1], g[2]])
    alto = fuzz.trimf(u, [g[1], 1.0,  1.0])
    x = np.clip(x, 0.0, 1.0)
    return (
        np.interp(x, u, baixo),
        np.interp(x, u, medio),
        np.interp(x, u, alto),
    )


def avaliar_fuzzy(params, nn_arr, chol_arr, thalach_arr, oldpeak_arr):
    p = np.clip(params[:12], 0.01, 0.99)
    grupos = [np.sort(p[i:i+3]) for i in range(0, 12, 3)]

    nn_b,  nn_m,  nn_a = _pertinencias(nn_arr,      grupos[0])
    ch_b,  ch_m,  ch_a = _pertinencias(chol_arr,    grupos[1])
    th_b,  th_m,  th_a = _pertinencias(thalach_arr, grupos[2])
    st_b,  st_m,  st_a = _pertinencias(oldpeak_arr, grupos[3])

    forca_alto = np.maximum.reduce([
        nn_a,
        np.minimum(nn_m, ch_a),
        np.minimum(nn_m, th_b),
        np.minimum(nn_m, st_a),
        np.minimum.reduce([nn_b, ch_a, th_b, st_a]),
    ])
    forca_medio = np.maximum.reduce([
        np.minimum.reduce([nn_m, ch_m, th_m, st_m]),
        np.minimum.reduce([nn_b, ch_a, th_m, st_m]),
        np.minimum.reduce([nn_b, ch_m, th_b, st_m]),
        np.minimum.reduce([nn_b, ch_m, th_m, st_a]),
    ])
    forca_baixo = np.maximum.reduce([
        np.minimum.reduce([nn_b, ch_b, th_a, st_b]),
        np.minimum.reduce([nn_b, ch_m, th_a, st_b]),
        np.minimum.reduce([nn_b, ch_b, th_m, st_b]),
    ])

    u_s = np.linspace(0, 100, 101)
    mf_b = fuzz.trimf(u_s, [0,  0,  35])
    mf_m = fuzz.trimf(u_s, [25, 50, 75])
    mf_a = fuzz.trimf(u_s, [65, 100, 100])

    agg = np.maximum(
        np.maximum(
            np.minimum(forca_baixo[:, None], mf_b[None, :]),
            np.minimum(forca_medio[:, None], mf_m[None, :]),
        ),
        np.minimum(forca_alto[:, None], mf_a[None, :]),
    )

    numer = (agg * u_s[None, :]).sum(axis=1)
    denom = agg.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        risco = np.where(denom > 1e-9, numer / denom, 50.0)
    return np.clip(risco, 0.0, 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# PASSO 5: ALGORITMO GENETICO
# ═══════════════════════════════════════════════════════════════════════════

def otimizar_ga(nn_val, chol_val, thalach_val, oldpeak_val, y_val,
                geracoes=40, populacao=30):

    def fitness(ga_inst, sol, sol_idx):
        saidas = avaliar_fuzzy(
            sol[:12], nn_val, chol_val, thalach_val, oldpeak_val)
        preds = (saidas >= sol[12]).astype(int)
        return f1_score(y_val, preds, zero_division=0)

    gene_space = (
        [{"low": 0.01, "high": 0.99}] * 12 +
        [{"low": 5.0,  "high": 95.0}]
    )
    ga = pygad.GA(
        num_generations=geracoes,
        num_parents_mating=8,
        fitness_func=fitness,
        sol_per_pop=populacao,
        num_genes=13,
        gene_space=gene_space,
        parent_selection_type="rank",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20,
        random_seed=42,
        suppress_warnings=True,
    )
    ga.run()
    sol, fit, _ = ga.best_solution()
    return sol, fit, ga


# ═══════════════════════════════════════════════════════════════════════════
# PASSO 6: RELATORIO
# ═══════════════════════════════════════════════════════════════════════════

def relatorio(nome, y_true, y_pred, y_prob):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true,    y_pred, zero_division=0)
    f1 = f1_score(y_true,        y_pred, zero_division=0)
    auc = roc_auc_score(y_true,   y_prob)
    acc = accuracy_score(y_true,  y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n  -- {nome} --")
    print(f"  Accuracy      : {acc:.3f}")
    print(f"  Precisao      : {p:.3f}")
    print(f"  Recall        : {r:.3f}")
    print(f"  F1-score      : {f1:.3f}")
    print(f"  AUC-ROC       : {auc:.3f}")
    print(f"  Especificidade: {spec:.3f}")
    print(f"  Matriz de confusao:")
    print(f"           Predito Saudavel  Predito Doenca")
    print(f"  Real Sau.    {cm[0, 0]:>5}           {cm[0, 1]:>5}")
    print(f"  Real Doe.    {cm[1, 0]:>5}           {cm[1, 1]:>5}")
    return [p, r, f1, auc], cm


# ═══════════════════════════════════════════════════════════════════════════
# PASSO 7: GRAFICOS
# ═══════════════════════════════════════════════════════════════════════════

def gerar_graficos(ga, m_nn, m_hibrido, cm_nn, cm_hibrido, params_mf):
    azul = "#1e40af"
    claro = "#93c5fd"
    cinza = "#94a3b8"
    verde = "#16a34a"
    laranja = "#b45309"

    # Convergencia GA
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ga.best_solutions_fitness, color=azul, linewidth=2,
            marker="o", markersize=3)
    ax.fill_between(range(len(ga.best_solutions_fitness)),
                    ga.best_solutions_fitness, alpha=0.15, color=claro)
    ax.set_xlabel("Geracao", fontsize=12)
    ax.set_ylabel("Melhor F1 (validacao)", fontsize=12)
    ax.set_title("Convergencia do Algoritmo Genetico",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_convergencia_ga.png", dpi=150)
    plt.close(fig)

    # Comparacao de metricas
    labels = ["Precisao", "Recall", "F1", "AUC-ROC"]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, m_nn,      w,
                label="Apenas Rede Neural",  color=cinza, edgecolor="white")
    b2 = ax.bar(x + w/2, m_hibrido, w,
                label="Hibrido Fuzzy-NN-GA", color=azul,  edgecolor="white")
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Comparacao de Desempenho - Conjunto de Teste",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_comparacao_metricas.png", dpi=150)
    plt.close(fig)

    # Funcoes de pertinencia
    p = np.clip(params_mf, 0.01, 0.99)
    grupos = [np.sort(p[i:i+3]) for i in range(0, 12, 3)]
    nomes = ["nn_score", "Colesterol (norm.)",
             "Thalach (norm.)", "Oldpeak (norm.)"]
    cores = [azul, verde, laranja]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, nome, g in zip(axes.flatten(), nomes, grupos):
        u = np.linspace(0, 1, 200)
        u_s = np.linspace(0, 1, 101)
        baixo = np.interp(u, u_s, fuzz.trimf(u_s, [0,    0,    g[1]]))
        medio = np.interp(u, u_s, fuzz.trimf(u_s, [g[0], g[1], g[2]]))
        alto = np.interp(u, u_s, fuzz.trimf(u_s, [g[1], 1,    1]))
        for mf, cor in zip([baixo, medio, alto], cores):
            ax.fill_between(u, mf, alpha=0.20, color=cor)
        ax.plot(u, baixo, color=cores[0], lw=2, label="Baixo")
        ax.plot(u, medio, color=cores[1], lw=2, label="Medio")
        ax.plot(u, alto,  color=cores[2], lw=2, label="Alto")
        ax.set_title(nome, fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Funcoes de Pertinencia - Parametros Otimizados pelo GA",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3_funcoes_pertinencia.png", dpi=150)
    plt.close(fig)

    # Matrizes de confusao
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm_data, titulo in zip(
        axes,
        [cm_nn, cm_hibrido],
        ["Apenas Rede Neural", "Hibrido Fuzzy-NN-GA"]
    ):
        cm = np.array(cm_data)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Saudavel", "Doenca"])
        ax.set_yticklabels(["Saudavel", "Doenca"])
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(titulo, fontweight="bold")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]),
                        ha="center", va="center", fontsize=14,
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
    fig.suptitle("Matrizes de Confusao - Conjunto de Teste",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "4_matrizes_confusao.png", dpi=150)
    plt.close(fig)

    print(f"  Graficos salvos em: {OUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# PASSO 8: PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print(" SISTEMA HIBRIDO FUZZY-NEURO-GENETICO - DOENCA CARDIACA")
    print("=" * 70)

    print(f"\n  Buscando CSV em: {CSV_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {CSV_PATH}\n"
            f"Coloque o heart_cleveland.csv na pasta: {BASE_DIR}"
        )

    print("\n[PASSO 1] Carregando dataset...")
    df = carregar_dataset()

    print("\n[PASSO 2] Preparando dados...")
    X_tr_s, X_val_s, X_te_s, y_tr, y_val, y_te, feats = preparar_dados(df)

    print("\n[PASSO 3] Treinando rede neural (MLP 24->12, ReLU, Adam)...")
    mlp = treinar_rede_neural(X_tr_s, y_tr)
    nn_val = mlp.predict_proba(X_val_s)[:, 1]
    nn_te = mlp.predict_proba(X_te_s)[:, 1]

    idx_chol = feats.index("chol")
    idx_thalach = feats.index("thalach")
    idx_oldpeak = feats.index("oldpeak")

    chol_val = X_val_s[:, idx_chol]
    thalach_val = X_val_s[:, idx_thalach]
    oldpeak_val = X_val_s[:, idx_oldpeak]

    chol_te = X_te_s[:, idx_chol]
    thalach_te = X_te_s[:, idx_thalach]
    oldpeak_te = X_te_s[:, idx_oldpeak]

    preds_nn = (nn_te >= 0.5).astype(int)

    print("\n[PASSO 4] Otimizando sistema fuzzy com algoritmo genetico...")
    print("  Cromossomo: 12 parametros fuzzy + 1 limiar | Fitness: F1 na validacao")

    sol, fit_val, ga = otimizar_ga(
        nn_val, chol_val, thalach_val, oldpeak_val, y_val
    )
    params_mf = sol[:12]
    limiar = sol[12]

    print(f"  Melhor F1 (validacao): {fit_val:.3f}")
    print(f"  Limiar de decisao GA : {limiar:.1f} (escala 0-100)")

    print("\n[PASSO 5] Avaliacao final no conjunto de teste...")
    risco_te = avaliar_fuzzy(params_mf, nn_te, chol_te, thalach_te, oldpeak_te)
    preds_hibrido = (risco_te >= limiar).astype(int)
    prob_hibrido = risco_te / 100.0

    m_nn,      cm_nn = relatorio(
        "Apenas Rede Neural (limiar 0.5)", y_te, preds_nn,      nn_te)
    m_hibrido, cm_hibrido = relatorio(
        "Hibrido Fuzzy-NN-GA",             y_te, preds_hibrido, prob_hibrido)

    print("\n[INTERPRETACAO CLINICA]")
    print("  Recall mais alto no hibrido = mais doentes identificados corretamente.")
    print("  Falso negativo e especialmente grave em diagnostico medico.")
    print("  O fuzzy torna o resultado explicavel via regras linguisticas.")
    print("  Ex: nn_score ALTO + oldpeak ALTO + thalach BAIXO -> risco ALTO")

    print("\n[PASSO 6] Gerando graficos...")
    gerar_graficos(ga, m_nn, m_hibrido, cm_nn, cm_hibrido, params_mf)

    print("\n" + "=" * 70)
    print(" CONCLUIDO")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
