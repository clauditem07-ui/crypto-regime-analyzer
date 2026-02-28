# 🔬 Crypto Market Regime Analyzer — HMM Edition

Aplicación interactiva en Python/Streamlit para detección de regímenes de mercado en criptoactivos usando **Modelos Ocultos de Markov (HMM)**.

## ⚡ Quick Start

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar la aplicación
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`

## 🧠 ¿Cómo funciona?

El modelo HMM (Gaussian Hidden Markov Model) detecta **estados latentes** del mercado analizando:

| Feature | Descripción |
|---------|-------------|
| **Retornos logarítmicos** | Dirección y magnitud del movimiento de precio |
| **RSI normalizado** | Momentum (sobrecompra/sobreventa) |
| **Volumen relativo** | Actividad vs. promedio de 20 períodos |
| **Volatilidad** | Desviación estándar rolling de retornos |

### Regímenes detectados

Según la configuración (3-7 regímenes), el modelo clasifica cada período en estados como:

- 💀 **Crash** — Caída severa con alta volatilidad
- 🐻 **Bear** — Tendencia bajista sostenida
- ➡️ **Neutral** — Consolidación/lateralización  
- 📈 **Bull** — Tendencia alcista
- 🚀 **Euphoria** — Rally fuerte con alta confianza

### Optimización anti-sobreajuste

- **Múltiples inicializaciones** (10-30 seeds aleatorios) para evitar óptimos locales
- **Covarianza Full** para capturar correlaciones entre features
- **Métricas AIC/BIC** para evaluar complejidad del modelo
- **Sorting automático** de regímenes por retorno medio (bear → bull)

## 📊 Activos disponibles

BTC, ETH, SOL, DOT, LINK, AVAX, ADA, XRP, HBAR, TAO

## 🔧 Configuración

- **Timeframes**: 1h, 4h, 1d
- **Regímenes**: 3-7 (slider)
- **Features**: Toggle individual de RSI, Volumen, Volatilidad
- **Rango de fechas**: Personalizable

## 📈 Outputs

1. **Gráfico de precio** coloreado por régimen detectado
2. **Panel de probabilidades** del régimen actual con nivel de confianza
3. **RSI overlay** con zonas de sobrecompra/sobreventa
4. **Matriz de transición** entre regímenes
5. **Señal de trading** (semáforo) basada en régimen actual
6. **Estadísticas** por régimen (retorno medio, volatilidad, RSI, volumen)

## ⚠️ Disclaimer

Esta herramienta es para análisis y educación. No constituye asesoramiento financiero. Los modelos estadísticos tienen limitaciones inherentes y el rendimiento pasado no garantiza resultados futuros.
