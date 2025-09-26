"""
Earnings Sentiment Analyzer - Tier 1 Feature
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.earnings import analyze_earnings_transcript, compare_earnings_quarters
from core.auth import is_pro_feature, show_pro_upgrade_prompt

def show_earnings_sentiment():
    """Display the earnings sentiment analyzer page"""
    
    st.title("📰 Earnings Sentiment Analyzer")
    st.markdown("AI-powered analysis of earnings call transcripts")
    
    # Check if user has pro access
    if not is_pro_feature():
        show_pro_upgrade_prompt("Earnings Sentiment Analysis")
        return
    
    # Sidebar for input options
    with st.sidebar:
        st.header("Analysis Options")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Single Transcript", "Quarter Comparison", "Batch Analysis"]
        )
        
        if analysis_type == "Single Transcript":
            st.subheader("Transcript Input")
            
            input_method = st.radio(
                "Input Method",
                ["Paste Text", "Upload File", "Sample Transcript"]
            )
            
            if input_method == "Paste Text":
                transcript_text = st.text_area(
                    "Earnings Transcript",
                    height=300,
                    placeholder="Paste the earnings call transcript here..."
                )
            elif input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload Transcript File",
                    type=['txt', 'pdf', 'docx'],
                    help="Upload a text file containing the earnings transcript"
                )
                if uploaded_file:
                    # For now, just show file info - in production, you'd parse the file
                    st.info(f"File uploaded: {uploaded_file.name}")
                    transcript_text = "Sample transcript text for demonstration..."
                else:
                    transcript_text = ""
            else:  # Sample transcript
                transcript_text = """
                Good morning, everyone. Thank you for joining our Q3 earnings call. 
                We're pleased to report strong quarterly results with revenue growth of 15% year-over-year. 
                Our core business segments continue to perform well, and we're seeing positive momentum 
                in our key markets. However, we face some headwinds in the current economic environment 
                that may impact our near-term outlook. We remain confident in our long-term strategy 
                and are committed to delivering value to our shareholders.
                """
                st.text_area("Sample Transcript", value=transcript_text, height=200, disabled=True)
        
        elif analysis_type == "Quarter Comparison":
            st.subheader("Quarter Comparison")
            
            num_quarters = st.number_input(
                "Number of Quarters", 
                min_value=2, 
                max_value=4, 
                value=2
            )
            
            st.write("Enter transcripts for each quarter:")
            quarter_transcripts = []
            
            for i in range(num_quarters):
                quarter_name = f"Q{4-num_quarters+i+1}"
                transcript = st.text_area(
                    f"{quarter_name} Transcript",
                    height=150,
                    key=f"quarter_{i}",
                    placeholder=f"Paste {quarter_name} earnings transcript..."
                )
                quarter_transcripts.append({
                    'quarter': quarter_name,
                    'transcript': transcript
                })
        
        else:  # Batch Analysis
            st.subheader("Batch Analysis")
            st.info("Batch analysis allows you to analyze multiple transcripts at once. This feature is coming soon!")
    
    # Main content area
    if st.button("Analyze Sentiment", type="primary"):
        
        if analysis_type == "Single Transcript":
            if not transcript_text or len(transcript_text.strip()) < 100:
                st.error("Please provide a valid earnings transcript (at least 100 characters)")
                return
            
            # Analyze the transcript
            with st.spinner("Analyzing sentiment..."):
                results = analyze_earnings_transcript(transcript_text)
            
            if 'error' in results:
                st.error(results['error'])
                return
            
            # Display results
            st.subheader("Sentiment Analysis Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Sentiment", results['sentiment_label'])
            
            with col2:
                st.metric("Sentiment Score", f"{results['sentiment_score']:.3f}")
            
            with col3:
                st.metric("Confidence", f"{results['analysis_confidence']:.1f}%")
            
            with col4:
                st.metric("Word Count", f"{results['word_count']:,}")
            
            # Sentiment breakdown
            st.subheader("Sentiment Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Positive Words", results['positive_words'])
            
            with col2:
                st.metric("Negative Words", results['negative_words'])
            
            # Sentiment score visualization
            fig = go.Figure()
            
            # Create a gauge chart for sentiment score
            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = results['sentiment_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Score"},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.2], 'color': "lightgray"},
                        {'range': [-0.2, 0.2], 'color': "yellow"},
                        {'range': [0.2, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key phrases
            st.subheader("Key Phrases")
            
            if results['key_phrases']:
                for i, phrase in enumerate(results['key_phrases'], 1):
                    st.write(f"{i}. {phrase}")
            else:
                st.info("No key phrases extracted from the transcript")
            
            # Summary
            st.subheader("Analysis Summary")
            st.write(results['summary'])
            
            # Detailed analysis
            st.subheader("Detailed Analysis")
            
            analysis_data = {
                "Metric": ["TextBlob Polarity", "Custom Sentiment", "Weighted Sentiment", "Subjectivity"],
                "Value": [
                    f"{results['textblob_polarity']:.3f}",
                    f"{results['custom_sentiment']:.3f}",
                    f"{results['sentiment_score']:.3f}",
                    f"{results['textblob_subjectivity']:.3f}"
                ],
                "Description": [
                    "General sentiment polarity from TextBlob",
                    "Financial-specific sentiment analysis",
                    "Weighted combination of both methods",
                    "How subjective vs objective the text is"
                ]
            }
            
            analysis_df = pd.DataFrame(analysis_data)
            st.dataframe(analysis_df, use_container_width=True)
            
            # Download results
            st.subheader("Download Results")
            
            # Create results DataFrame
            results_data = {
                'Metric': ['Sentiment Score', 'Sentiment Label', 'Positive Words', 'Negative Words', 'Word Count', 'Confidence'],
                'Value': [
                    results['sentiment_score'],
                    results['sentiment_label'],
                    results['positive_words'],
                    results['negative_words'],
                    results['word_count'],
                    results['analysis_confidence']
                ]
            }
            
            results_df = pd.DataFrame(results_data)
            csv = results_df.to_csv(index=False)
            
            st.download_button(
                label="Download Analysis Results (CSV)",
                data=csv,
                file_name=f"earnings_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        elif analysis_type == "Quarter Comparison":
            # Check if all quarters have transcripts
            valid_transcripts = [qt for qt in quarter_transcripts if qt['transcript'].strip()]
            
            if len(valid_transcripts) < 2:
                st.error("Please provide transcripts for at least 2 quarters")
                return
            
            # Analyze each quarter
            with st.spinner("Analyzing quarterly sentiment..."):
                results = compare_earnings_quarters(valid_transcripts)
            
            if 'error' in results:
                st.error(results['error'])
                return
            
            # Display results
            st.subheader("Quarterly Sentiment Comparison")
            
            # Trend analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Latest Sentiment", results['latest_sentiment'])
            
            with col2:
                st.metric("Trend", results['trend'].title())
            
            with col3:
                st.metric("Sentiment Change", f"{results['sentiment_change']:.3f}")
            
            # Quarterly results table
            st.subheader("Quarterly Results")
            
            quarterly_df = pd.DataFrame(results['quarterly_results'])
            st.dataframe(quarterly_df, use_container_width=True)
            
            # Sentiment trend chart
            fig = px.line(
                quarterly_df, 
                x='quarter', 
                y='sentiment_score',
                title="Sentiment Trend Over Quarters",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Quarter",
                yaxis_title="Sentiment Score"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Key phrases comparison
            st.subheader("Key Phrases by Quarter")
            
            for result in results['quarterly_results']:
                st.write(f"**{result['quarter']}:**")
                if result['key_phrases']:
                    for phrase in result['key_phrases']:
                        st.write(f"• {phrase}")
                else:
                    st.write("No key phrases extracted")
                st.write("---")
            
            # Download results
            csv = quarterly_df.to_csv(index=False)
            st.download_button(
                label="Download Quarterly Comparison (CSV)",
                data=csv,
                file_name=f"quarterly_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        else:  # Batch Analysis
            st.info("Batch analysis feature is coming soon! This will allow you to analyze multiple transcripts simultaneously.")
    
    # Usage tips
    st.subheader("Usage Tips")
    
    tips = [
        "Paste the complete earnings call transcript for best results",
        "Include both management commentary and Q&A sections",
        "Longer transcripts generally provide more accurate sentiment analysis",
        "Compare sentiment across multiple quarters to identify trends",
        "Look for changes in key phrases and sentiment over time",
        "Consider the overall market context when interpreting results"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.write(f"{i}. {tip}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>⚠️ <strong>Disclaimer:</strong> This sentiment analysis is for educational purposes only. Not investment advice.</p>
        <p>Sentiment analysis results should be used as one factor in your investment research.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_earnings_sentiment()
