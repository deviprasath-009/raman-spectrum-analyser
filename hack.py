import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import os
import streamlit as st
import json
import pubchempy as pcp
import requests
import genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import time
from typing import Dict, List, Union

# ----------- OpenRouter AI Integration -------------
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))

def generate_ai_summary_openrouter(compound_name, group):
    prompt = (
        f"Provide a detailed description of the compound '{compound_name}' (group: {group}), "
        f"highlighting its chemical significance, common applications, and relevance in Raman spectroscopy. "
        f"Include information about its functional groups and any notable properties. "
        f"Format the response with clear sections and bullet points."
    )

    # Setup API endpoint and headers
    url = "https://api.openrouter.com/v1/ai-summary"  # Replace with actual API endpoint
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Setup payload with the prompt
    payload = {
        "prompt": prompt,
        "max_tokens": 2000,  # You can adjust the token length
    }

    # Send POST request to the API
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Will raise an exception for a bad status
        ai_summary = response.json().get("text", "")
        return ai_summary
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating summary: {e}")
        return None

# ----------- Peak Fitting Functions -------------
def lorentzian(x, a, x0, gamma):
    return a * gamma**2 / ((x - x0)**2 + gamma**2)

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

# ----------- Molecular Identifier Class -------------
class MolecularIdentifier:
    def __init__(self, tolerance: float = 50, min_matches: int = 1):
        self.tolerance = tolerance
        self.min_matches = min_matches

    def identify(self, peaks: List[float], database: Dict) -> List[Dict]:
        all_matches = []
        for category, compounds in database.items():
            for compound in compounds:
                match_count = 0
                ref_peaks = compound.get("Peaks", [])
                for ref_peak in ref_peaks:
                    ref_wavenumber = ref_peak.get("Wavenumber")
                    if any(abs(peak - ref_wavenumber) <= self.tolerance for peak in peaks):
                        match_count += 1
                if match_count >= self.min_matches:
                    compound_name = compound.get("Name")
                    try:
                        compound_data = pcp.get_compounds(compound_name, 'name')
                        pubchem_link = ''
                        if compound_data:
                            pubchem_link = f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound_data[0].cid}"
                        all_matches.append({
                            "compound": compound_name,
                            "group": category,
                            "matched_peaks": match_count,
                            "pubchem_link": pubchem_link
                        })
                    except Exception as e:
                        st.warning(f"PubChem lookup failed for {compound_name}: {str(e)}")
        return sorted(all_matches, key=lambda x: x["matched_peaks"], reverse=True)

# ----------- Raman Analyzer Class (partial) -------------
# ----------- Raman Analyzer Class -------------
class RamanAnalyzer(BaseEstimator, ClassifierMixin):
    def __init__(self, prominence_factor: float = 0.5, min_distance: int = 10, 
                 model_path: str = None, json_paths: Union[str, List[str]] = None):
        self.prominence_factor = prominence_factor
        self.min_distance = min_distance
        self.model_path = model_path
        self.json_paths = json_paths if isinstance(json_paths, list) else [json_paths] if json_paths else []
        self.identifier = MolecularIdentifier()
        self.database = self._load_all_databases()

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
        ])
        if model_path:
            self.load_model(model_path)

    def _load_all_databases(self) -> Dict:
        """Load and merge multiple JSON databases"""
        merged_db = {}
        for path in self.json_paths:
            try:
                if path:  # Skip None or empty paths
                    with open(path, 'r', encoding='utf-8') as f:
                        db = json.load(f)
                        # Merge databases by category
                        for category, compounds in db.items():
                            if category in merged_db:
                                merged_db[category].extend(compounds)
                            else:
                                merged_db[category] = compounds
            except FileNotFoundError:
                st.warning(f"Database file not found: {path}")
            except json.JSONDecodeError:
                st.error(f"Failed to decode JSON file: {path}")
            except Exception as e:
                st.error(f"Error loading database {path}: {str(e)}")
        
        if not merged_db:
            st.warning("No valid databases loaded. Using empty database.")
        return merged_db

    def add_database(self, db_data: Dict):
        """Add in-memory database data"""
        for category, compounds in db_data.items():
            if category in self.database:
                self.database[category].extend(compounds)
            else:
                self.database[category] = compounds

    def load_metadata_database(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            st.warning("Raman metadata database not found.")
            return {}
        except json.JSONDecodeError:
            st.error("Failed to decode JSON file. Please check the formatting.")
            return {}

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def detect_peaks(self, wavenumbers, intensities):
        intensities_normalized = 5 * (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        prominence = np.std(intensities_normalized) * self.prominence_factor
        peaks, _ = find_peaks(intensities_normalized, prominence=prominence, distance=self.min_distance)
        return wavenumbers[peaks], intensities[peaks]

    def _create_feature_vector(self, wavenumbers, intensities):
        bins = np.arange(100, 4001, 100)
        binned_intensity = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            start, end = bins[i], bins[i + 1]
            mask = (wavenumbers >= start) & (wavenumbers < end)
            if np.any(mask):
                binned_intensity[i] = np.max(intensities[mask])
        return binned_intensity

    def extract_peak_metadata(self, wavenumbers, intensities, peaks):
        peak_info = []
        for peak in peaks:
            idx = np.argmin(np.abs(wavenumbers - peak))
            window = (wavenumbers > peak - 20) & (wavenumbers < peak + 20)
            x = wavenumbers[window]
            y = intensities[window]

            shape_type = "Unknown"
            fwhm = np.nan
            asymmetry = np.nan

            try:
                popt_l, _ = curve_fit(lorentzian, x, y, p0=[max(y), peak, 5])
                lor_res = np.sum((lorentzian(x, *popt_l) - y)**2)
                popt_g, _ = curve_fit(gaussian, x, y, p0=[max(y), peak, 5])
                gauss_res = np.sum((gaussian(x, *popt_g) - y)**2)
                if lor_res < gauss_res:
                    shape_type = "Lorentzian"
                    fwhm = 2 * popt_l[2]
                else:
                    shape_type = "Gaussian"
                    fwhm = 2.355 * popt_g[2]

                left_idx = np.argmax(y >= max(y) / 2)
                right_idx = len(y) - np.argmax(y[::-1] >= max(y) / 2)
                if right_idx > left_idx:
                    asymmetry = (x[right_idx - 1] - peak) / (peak - x[left_idx])

            except Exception:
                pass

            functional_group = ""
            for group, compounds in self.database.items():
                for compound in compounds:
                    for peak_entry in compound.get("Peaks", []):
                        if abs(peak_entry.get("Wavenumber", 0) - peak) <= self.identifier.tolerance:
                            functional_group = peak_entry.get("Assignment", "")
                            break
                    if functional_group:
                        break
                if functional_group:
                    break

            peak_info.append({
                "Wavenumber": peak,
                "Intensity": intensities[idx],
                "FWHM": fwhm,
                "Asymmetry": asymmetry,
                "Shape Type": shape_type,
                "Functional Group": functional_group or "Unassigned"
            })
        return pd.DataFrame(peak_info)

    def analyze_spectrum(self, wavenumbers, intensities):
        peaks_wavenum, _ = self.detect_peaks(wavenumbers, intensities)
        feature_vector = self._create_feature_vector(wavenumbers, intensities)
        compound_suggestions = self.identifier.identify(peaks_wavenum, self.database)
        return {
            'peaks': peaks_wavenum,
            'feature_vector': feature_vector,
            'compound_suggestions': compound_suggestions  # Fixed this line
        }

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        self.model = joblib.load(filepath)
        return self

    def visualize_analysis(self, wavenumbers, intensities, results):
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(wavenumbers, intensities, label='Raman Spectrum', linewidth=1)
        if results.get('peaks') is not None:
            for peak in results['peaks']:
                idx = np.argmin(np.abs(wavenumbers - peak))
                ax.scatter(peak, intensities[idx], color='red', zorder=5)
        ax.set_title("Raman Spectrum Analysis")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.grid(True)
        ax.legend()
        ax.invert_xaxis()
        plt.tight_layout()
        return fig

    def generate_ai_summary(self, compound_name, group):
        prompt = (
            f"Provide a detailed description of the compound '{compound_name}' (group: {group}), "
            f"highlighting its chemical significance, common applications, and relevance in Raman spectroscopy. "
            f"Include information about its functional groups and any notable properties. "
            f"Format the response with clear sections and bullet points."
        )
        try:
            model = genai.GenerativeModel('gemini-1.0-pro-001')
            response = model.generate_content(prompt)
            return response.text if response.text else "No summary generated"
        except Exception as e:
            return f"An error occurred while generating the summary: {e}"

    def generate_pdf_report(self, peak_metadata_df: pd.DataFrame, 
                          compound_suggestions: List[Dict], 
                          spectrum_fig: plt.Figure, 
                          output_path: str = "raman_report.pdf") -> str:
        """Generate PDF report with proper file handling"""
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph("<b>Raman Spectrum Analysis Report</b>", styles['Title']))
        elements.append(Spacer(1, 12))

        # Compound info
        if compound_suggestions:
            top_compound = compound_suggestions[0]
            compound_name = top_compound["compound"]
            group = top_compound["group"]
            pubchem_link = top_compound.get("pubchem_link", "N/A")

            try:
                summary_text = self.generate_ai_summary(compound_name, group)
                elements.append(Paragraph(f"<b>Top Compound Suggested:</b> {compound_name} ({group})", styles['Heading2']))
                elements.append(Paragraph(f"<b>PubChem Link:</b> <a href='{pubchem_link}'>{pubchem_link}</a>", styles['Normal']))
                elements.append(Paragraph("<b>AI-Generated Summary:</b>", styles['Heading3']))
                elements.append(Paragraph(summary_text, styles['Normal']))
                elements.append(Spacer(1, 12))
            except Exception as e:
                st.error(f"Failed to generate AI summary: {str(e)}")

        # Peak table
        if not peak_metadata_df.empty:
            elements.append(Paragraph("<b>Peak Interpretation:</b>", styles['Heading2']))
            table_data = [peak_metadata_df.columns.tolist()] + peak_metadata_df.astype(str).values.tolist()
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

        # Spectrum plot with proper file handling
        temp_img_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                temp_img_path = tmpfile.name
                spectrum_fig.savefig(temp_img_path, dpi=300, bbox_inches='tight')
                elements.append(Paragraph("<b>Spectrum Plot:</b>", styles['Heading2']))
                elements.append(RLImage(temp_img_path, width=500, height=300))

            doc.build(elements)
        finally:
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except PermissionError:
                    time.sleep(0.5)  # Wait and retry
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass  # Give up if still locked

        return output_path

# ----------- Streamlit App Interface -------------
@st.cache_resource
def get_analyzer(json_paths: List[str] = None) -> RamanAnalyzer:
    """Create analyzer with cached instance"""
    default_paths = ["C:\\Users\\dpras\\Downloads\\up.json"]  # Add your default paths here
    paths = json_paths if json_paths else default_paths
    return RamanAnalyzer(json_paths=paths)

def main():
    st.set_page_config(page_title="Raman Spectrum Analyzer", layout="wide")
    st.title("🔬 Raman Spectrum Analyzer with AI")

    # Database configuration
    st.sidebar.header("Database Configuration")
    uploaded_dbs = st.sidebar.file_uploader(
        "Upload Custom Database (JSON)", 
        type=['json'],
        accept_multiple_files=True
    )

    # Main file upload
    uploaded = st.file_uploader(
        "Upload your Raman CSV (wavenumber, intensity)", 
        type=['csv']
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Uploaded Data Preview", df.head())

            wavenumbers = df.iloc[:, 0].values
            intensities = df.iloc[:, 1].values

            # Initialize analyzer
            analyzer = get_analyzer()

            # Add uploaded databases
            if uploaded_dbs:
                for uploaded_db in uploaded_dbs:
                    try:
                        db_data = json.load(uploaded_db)
                        analyzer.add_database(db_data)
                        st.sidebar.success(f"Loaded: {uploaded_db.name}")
                    except Exception as e:
                        st.sidebar.error(f"Error loading {uploaded_db.name}: {str(e)}")

            # Analyze spectrum
            results = analyzer.analyze_spectrum(wavenumbers, intensities)  # This should now work

            # Display results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Peak Interpretation")
                peak_metadata = analyzer.extract_peak_metadata(wavenumbers, intensities, results['peaks'])
                st.dataframe(peak_metadata)

                st.subheader("Compound Suggestions")
                if not results['compound_suggestions']:
                    st.info("No compound suggestions found.")
                else:
                    compound_df = pd.DataFrame(results['compound_suggestions'])
                    if 'pubchem_link' in compound_df.columns:
                        compound_df['PubChem Link'] = compound_df['pubchem_link'].apply(
                            lambda x: f'<a href="{x}">Link</a>' if x else "N/A"
                        )
                        compound_df.drop(columns=['pubchem_link'], inplace=True)
                    st.dataframe(compound_df, hide_index=True)

            with col2:
                st.subheader("Spectrum Visualization")
                fig = analyzer.visualize_analysis(wavenumbers, intensities, results)
                st.pyplot(fig)

                # PDF Report Generation
                if results.get('compound_suggestions'):
                    if st.button("📄 Generate PDF Report", type="primary"):
                        with st.spinner("Generating report..."):
                            report_path = analyzer.generate_pdf_report(
                                peak_metadata,
                                results['compound_suggestions'],
                                fig
                            )
                            with open(report_path, "rb") as f:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=f,
                                    file_name="raman_analysis_report.pdf",
                                    mime="application/pdf"
                                )
                            try:
                                os.remove(report_path)
                            except:
                                pass

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV file containing wavenumber and intensity data")

if __name__ == "__main__":
    main()