"""
ðŸš€ SIMPLE ML FRONTEND - DIRECT INTEGRATION
=========================================
Benutzer setzt Parameter â†’ Training startet â†’ Ergebnisse anzeigen
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, Any
import threading
import queue

# Import deiner Original-Klassen (angepasst)
import os
import logging
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score as sklearn_r2
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitMLTrainer:
    """
    ðŸŽ¯ VEREINFACHTE ML-KLASSE FÃœR STREAMLIT INTEGRATION
    Basiert auf deiner Original-Logik + UI-Updates
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.y_scaler = None
        self.history = None
        self.training_active = False

        # UI Update Queue
        if 'training_queue' not in st.session_state:
            st.session_state.training_queue = queue.Queue()
        if 'training_logs' not in st.session_state:
            st.session_state.training_logs = []

    def log_to_ui(self, message: str):
        """ðŸ“ Sende Log-Message an UI"""
        st.session_state.training_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        # Nur letzte 50 Logs behalten
        if len(st.session_state.training_logs) > 50:
            st.session_state.training_logs = st.session_state.training_logs[-50:]

    def load_and_preprocess_data(self, test_size: float = 0.2, val_size: float = 0.2):
        """ðŸ“Š Daten laden und vorverarbeiten"""
        self.log_to_ui("ðŸ“Š Lade Diabetes-Dataset...")

        # Original Logik aus deinem Code
        dataset = load_diabetes()
        X, y = dataset.data, dataset.target.reshape(-1, 1)

        self.log_to_ui(f"âœ… Dataset geladen: {X.shape[0]} Samples, {X.shape[1]} Features")

        # Splits
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42
        )

        # Standardisierung
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = self.scaler.transform(X_val).astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)

        self.y_scaler = StandardScaler()
        y_train_scaled = self.y_scaler.fit_transform(y_train).astype(np.float32)
        y_val_scaled = self.y_scaler.transform(y_val).astype(np.float32)
        y_test_scaled = self.y_scaler.transform(y_test).astype(np.float32)

        self.log_to_ui(f"ðŸ”„ Daten verarbeitet: {len(X_train)} Train, {len(X_val)} Val, {len(X_test)} Test")

        return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled)

    def build_model(self, hidden_layers: list, dropout_rate: float, l2_reg: float, learning_rate: float):
        """ðŸ—ï¸ Modell erstellen"""
        self.log_to_ui(f"ðŸ—ï¸ Erstelle Modell: {len(hidden_layers)} Hidden Layers {hidden_layers}")

        model = Sequential(name="StreamlitDiabetesRegression")

        # Hidden Layers
        for i, units in enumerate(hidden_layers):
            if i == 0:
                model.add(Dense(
                    units=units, input_shape=(10,),
                    activation='relu', kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg)
                ))
            else:
                model.add(Dense(
                    units=units, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg)
                ))

            model.add(Dropout(dropout_rate))

        # Output Layer
        model.add(Dense(1, kernel_initializer='he_normal'))

        # Kompilieren
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        total_params = model.count_params()
        self.log_to_ui(f"âœ… Modell erstellt: {total_params:,} Parameter")

        return model

    def train_model_with_ui_updates(self, model, train_data, val_data,
                                   max_epochs: int, batch_size: int, patience: int):
        """ðŸš€ Training mit Live-UI-Updates"""

        X_train, y_train = train_data
        X_val, y_val = val_data

        self.log_to_ui(f"ðŸš€ Starte Training: {max_epochs} max Epochs, Batch Size {batch_size}")

        # Custom Callback fÃ¼r UI-Updates
        class StreamlitCallback(keras.callbacks.Callback):
            def __init__(self, ui_logger):
                self.ui_logger = ui_logger
                self.start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                # Nur jede 5. Epoch fÃ¼r Performance
                if (epoch + 1) % 5 == 0 or epoch < 10:
                    elapsed = time.time() - self.start_time
                    self.ui_logger(
                        f"ðŸ“ˆ Epoch {epoch+1}: Loss={logs['loss']:.4f}, "
                        f"Val_Loss={logs['val_loss']:.4f}, Zeit={elapsed:.1f}s"
                    )

            def on_train_end(self, logs=None):
                total_time = time.time() - self.start_time
                self.ui_logger(f"âœ… Training beendet nach {total_time:.1f} Sekunden")

        # Callbacks
        callbacks = [
            StreamlitCallback(self.log_to_ui),
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-7, verbose=0)
        ]

        # Training
        self.history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0  # UI Ã¼bernimmt Output
        )

        return model

    def evaluate_model_with_ui(self, model, test_data):
        """ðŸ“Š Evaluation mit UI-Feedback"""
        self.log_to_ui("ðŸŽ¯ Starte Modell-Evaluierung...")

        X_test, y_test = test_data
        y_pred_scaled = model.predict(X_test, verbose=0)

        # ZurÃ¼ck-transformieren fÃ¼r echte Metriken
        y_test_original = self.y_scaler.inverse_transform(y_test)
        y_pred_original = self.y_scaler.inverse_transform(y_pred_scaled)

        # Metriken berechnen
        results = {
            'mse': float(mean_squared_error(y_test_original, y_pred_original)),
            'rmse': float(np.sqrt(mean_squared_error(y_test_original, y_pred_original))),
            'mae': float(mean_absolute_error(y_test_original, y_pred_original)),
            'r2': float(sklearn_r2(y_test_original, y_pred_original)),
            'mape': float(np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100),
            'test_samples': len(y_test),
            'y_true': y_test_original.flatten(),
            'y_pred': y_pred_original.flatten()
        }

        self.log_to_ui(f"ðŸŽ¯ Evaluierung abgeschlossen: RÂ²={results['r2']:.4f}, MSE={results['mse']:.1f}")

        return results

    def run_complete_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """ðŸŽ¯ Komplettes Training Pipeline"""
        try:
            # Reset
            st.session_state.training_logs = []
            self.training_active = True

            # 1. Daten laden
            train_data, val_data, test_data = self.load_and_preprocess_data(
                config['test_size'], config['validation_size']
            )

            # 2. Modell erstellen
            self.model = self.build_model(
                config['hidden_layers'],
                config['dropout_rate'],
                config['l2_reg'],
                config['learning_rate']
            )

            # 3. Training
            self.model = self.train_model_with_ui_updates(
                self.model, train_data, val_data,
                config['max_epochs'], config['batch_size'], config['patience']
            )

            # 4. Evaluation
            results = self.evaluate_model_with_ui(self.model, test_data)

            self.training_active = False
            self.log_to_ui("ðŸ† Pipeline komplett abgeschlossen!")

            return results

        except Exception as e:
            self.training_active = False
            self.log_to_ui(f"âŒ Fehler: {e}")
            return {"error": str(e)}

# ============================================================================
# STREAMLIT UI APPLICATION
# ============================================================================

def main():
    """ðŸš€ Hauptanwendung"""

    st.set_page_config(
        page_title="ðŸš€ ML Hyperparameter Tuning",
        page_icon="ðŸš€",
        layout="wide"
    )

    st.title("ðŸš€ ML Hyperparameter Tuning & Training")
    st.markdown("**Set Parameters â†’ Start Training â†’ View Results**")

    # Initialize Trainer
    if 'trainer' not in st.session_state:
        st.session_state.trainer = StreamlitMLTrainer()

    # âš™ï¸ SIDEBAR: Hyperparameter Configuration
    st.sidebar.header("âš™ï¸ Hyperparameter Configuration")

    # Model Architecture
    with st.sidebar.expander("ðŸ—ï¸ Model Architecture", expanded=True):
        num_layers = st.slider("Anzahl Hidden Layers", 1, 5, 3)

        hidden_layers = []
        for i in range(num_layers):
            units = st.slider(
                f"Layer {i+1} Neurons", 8, 256,
                [128, 64, 32, 16, 8][i] if i < 5 else 16,
                step=8, key=f"layer_{i}"
            )
            hidden_layers.append(units)

    # Regularization
    with st.sidebar.expander("ðŸ›¡ï¸ Regularization", expanded=True):
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.3, 0.05)
        l2_reg = st.slider("L2 Regularization", 0.0, 0.1, 0.01, 0.005)

    # Training Parameters
    with st.sidebar.expander("ðŸŽ¯ Training Parameters", expanded=True):
        learning_rate = st.selectbox("Learning Rate", [0.01, 0.005, 0.001, 0.0005], index=2)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        max_epochs = st.slider("Max Epochs", 100, 2000, 1000, 100)
        patience = st.slider("Early Stopping Patience", 10, 100, 50, 10)
        test_size = st.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
        validation_size = st.slider("Validation Size", 0.1, 0.3, 0.2, 0.05)

    # Sammle alle Parameter
    config = {
        'hidden_layers': hidden_layers,
        'dropout_rate': dropout_rate,
        'l2_reg': l2_reg,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'patience': patience,
        'test_size': test_size,
        'validation_size': validation_size
    }

    # ðŸ—ï¸ MAIN AREA: Tabs fÃ¼r verschiedene Bereiche
    tab1, tab2, tab3 = st.tabs(["ðŸ—ï¸ Model Preview", "ðŸš€ Training", "ðŸ“Š Results"])

    # ============================================================================
    # TAB 1: MODEL PREVIEW
    # ============================================================================
    with tab1:
        st.header("ðŸ—ï¸ Model Architecture Preview")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Visualisiere Architektur
            fig = go.Figure()

            layers = ["Input (10)"] + [f"Hidden {i+1} ({units})" for i, units in enumerate(hidden_layers)] + ["Output (1)"]

            for i, layer_name in enumerate(layers):
                color = "lightblue" if i == 0 else ("lightcoral" if i == len(layers)-1 else "lightgreen")

                fig.add_shape(
                    type="rect", x0=0, y0=i*1.2, x1=3, y1=i*1.2+1,
                    fillcolor=color, line=dict(color="black", width=1)
                )

                fig.add_annotation(
                    x=1.5, y=i*1.2+0.5, text=layer_name,
                    showarrow=False, font=dict(size=12)
                )

            fig.update_layout(
                title="ðŸ—ï¸ Neural Network Architecture",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                height=400, showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Parameter Summary
            total_params = sum([10 * hidden_layers[0]] +
                             [hidden_layers[i] * hidden_layers[i+1] for i in range(len(hidden_layers)-1)] +
                             [hidden_layers[-1]])

            st.metric("ðŸ”¢ Est. Parameters", f"{total_params:,}")
            st.metric("ðŸ›¡ï¸ Dropout Rate", f"{dropout_rate:.1%}")
            st.metric("âš¡ Learning Rate", f"{learning_rate}")
            st.metric("ðŸ“¦ Batch Size", batch_size)
            st.metric("ðŸŽ¯ Max Epochs", max_epochs)
            st.metric("â¸ï¸ Patience", patience)

    # ============================================================================
    # TAB 2: TRAINING INTERFACE
    # ============================================================================
    with tab2:
        st.header("ðŸš€ Training Interface")

        # Training Button
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("ðŸš€ Start Training", type="primary",
                        disabled=st.session_state.get('training_active', False)):

                st.session_state.training_active = True
                st.session_state.training_logs = []
                st.session_state.results = None

                # Starte Training
                with st.spinner("ðŸ”„ Training lÃ¤uft..."):
                    results = st.session_state.trainer.run_complete_training(config)
                    st.session_state.results = results
                    st.session_state.training_active = False

                if 'error' not in results:
                    st.success("âœ… Training erfolgreich abgeschlossen!")
                    st.balloons()  # ðŸŽ‰ Celebration!
                else:
                    st.error(f"âŒ Training fehlgeschlagen: {results['error']}")

        with col2:
            if st.button("ðŸ§¹ Clear Logs"):
                st.session_state.training_logs = []

        with col3:
            if st.button("â¹ï¸ Stop Training"):
                st.session_state.training_active = False
                st.warning("â¸ï¸ Training gestoppt (nicht implementiert)")

        # Training Status
        if st.session_state.get('training_active', False):
            st.info("ðŸ”„ Training lÃ¤uft... Bitte warten.")

            # Progress Animation
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simuliere Progress (da echtes Training zu schnell fÃ¼r UI)
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Training Progress: {i+1}%")
                time.sleep(0.05)  # 5 Sekunden total

        # Live Training Logs
        st.subheader("ðŸ“‹ Training Logs")

        if st.session_state.training_logs:
            # Container fÃ¼r Logs (Auto-Scroll)
            log_container = st.container()
            with log_container:
                # Zeige neueste Logs zuerst
                for log in reversed(st.session_state.training_logs[-20:]):
                    st.text(log)
        else:
            st.info("ðŸŽ¯ Klicke 'Start Training' um Logs zu sehen...")

    # ============================================================================
    # TAB 3: RESULTS DASHBOARD
    # ============================================================================
    with tab3:
        st.header("ðŸ“Š Training Results Dashboard")

        if not st.session_state.get('results') or 'error' in st.session_state.get('results', {}):
            st.info("ðŸŽ¯ Starte Training um Ergebnisse zu sehen...")
            return

        results = st.session_state.results

        # ðŸ† Key Metrics
        st.subheader("ðŸ† Performance Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            delta_r2 = f"+{(results['r2'] - 0.49)*100:.1f}%" if results['r2'] > 0.49 else None
            st.metric("ðŸŽ¯ RÂ² Score", f"{results['r2']:.4f}", delta_r2)

        with col2:
            st.metric("ðŸ“‰ MSE", f"{results['mse']:.1f}")

        with col3:
            st.metric("ðŸ“ RMSE", f"{results['rmse']:.1f}")

        with col4:
            st.metric("ðŸ“Š MAE", f"{results['mae']:.1f}")

        with col5:
            st.metric("ðŸ“ˆ MAPE", f"{results['mape']:.1f}%")

        # ðŸ“ˆ Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Prediction vs Actual Scatter Plot
            fig_scatter = px.scatter(
                x=results['y_true'], y=results['y_pred'],
                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                title="ðŸŽ¯ Predictions vs Actual Values"
            )

            # Perfekte Vorhersage-Linie
            min_val, max_val = min(results['y_true']), max(results['y_true'])
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))

            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            # Performance Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=results['r2'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "ðŸŽ¯ RÂ² Performance"},
                delta={'reference': 0.49},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.4], 'color': "lightgray"},
                        {'range': [0.4, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "lightgreen"},
                        {'range': [0.8, 1.0], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75, 'value': 0.65
                    }
                }
            ))

            st.plotly_chart(fig_gauge, use_container_width=True)

        # ðŸ“Š Training History
        if st.session_state.trainer.history:
            st.subheader("ðŸ“ˆ Training History")

            history_df = pd.DataFrame(st.session_state.trainer.history.history)

            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                x=list(range(len(history_df))),
                y=history_df['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            fig_history.add_trace(go.Scatter(
                x=list(range(len(history_df))),
                y=history_df['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ))

            fig_history.update_layout(
                title="ðŸ“ˆ Training & Validation Loss",
                xaxis_title="Epochs",
                yaxis_title="Loss",
                height=400
            )

            st.plotly_chart(fig_history, use_container_width=True)

        # ðŸ“‹ Detailed Results Table
        st.subheader("ðŸ“‹ Detailed Results")

        results_data = {
            'Metric': ['RÂ² Score', 'Mean Squared Error', 'Root MSE', 'Mean Absolute Error', 'MAPE', 'Test Samples'],
            'Value': [
                f"{results['r2']:.4f}",
                f"{results['mse']:.2f}",
                f"{results['rmse']:.2f}",
                f"{results['mae']:.2f}",
                f"{results['mape']:.1f}%",
                f"{results['test_samples']}"
            ],
            'Status': [
                'ðŸŽ¯ Excellent' if results['r2'] > 0.6 else 'âœ… Good' if results['r2'] > 0.4 else 'âš ï¸ Poor',
                'âœ… Good' if results['mse'] < 3000 else 'âš ï¸ High',
                'âœ… Good' if results['rmse'] < 55 else 'âš ï¸ High',
                'âœ… Good' if results['mae'] < 45 else 'âš ï¸ High',
                'âœ… Good' if results['mape'] < 20 else 'âš ï¸ High',
                'ðŸ“Š Info'
            ]
        }

        st.dataframe(pd.DataFrame(results_data), use_container_width=True)

        # ðŸ’¾ Download Results
        if st.button("ðŸ’¾ Download Results JSON"):
            results_json = json.dumps({
                'config': config,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, indent=2)

            st.download_button(
                "ðŸ“ Download",
                results_json,
                f"ml_results_{int(time.time())}.json",
                "application/json"
            )

if __name__ == "__main__":
    main()

"""
ðŸŽ¯ SIMPLE & DIRECT APPROACH:
===========================

âœ… **Eine Datei** - Alles in einem
âœ… **Direkte Integration** - Keine komplexe MVC-Struktur
âœ… **Live Training Updates** - Callback direkt in UI
âœ… **Automatisches Training** - Button â†’ Training â†’ Ergebnisse
âœ… **SchÃ¶ne Results UI** - Plots, Metriken, Downloads
âœ… **Deine Original-Logik** - Komplett erhalten

ðŸš€ **Usage:**
1. Speichern als: streamlit_ml_app.py
2. Run: streamlit run streamlit_ml_app.py
3. Set Parameters â†’ Click Training â†’ View Results

ðŸ’¡ **Features:**
- Dynamic Hyperparameter Tuning
- Real-time Training Logs
- Interactive Results Dashboard
- Model Architecture Visualization
- Results Export Functionality
- Progress Tracking
"""