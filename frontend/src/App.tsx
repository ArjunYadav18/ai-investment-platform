import React, { useState, useEffect, useRef } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import axios from 'axios';

ChartJS.register(ArcElement, Tooltip, Legend);

// --- Helper Icon Components ---
const BotIcon = ({ className = '' }) => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>;
const UserIcon = ({ className = '' }) => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>;
const SendIcon = ({ className = '' }) => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>;

// --- Static Mock Data (Briefs & Compliance) ---
const briefsData = {
    'AAPL': { company: 'Apple Inc. (AAPL)', date: 'October 26, 2025', sentiment: 'Positive', sentimentScore: '8.2/10', summary: 'Apple continues to demonstrate robust financial health driven by strong iPhone sales and significant growth in its Services division. The company\'s ecosystem remains a powerful competitive moat, fostering high customer loyalty and recurring revenue. Headwinds include geopolitical tensions impacting supply chains and increasing regulatory scrutiny globally.', risks: ['Intensifying competition in the smartphone market.', 'Dependence on Chinese manufacturing and consumer markets.', 'Antitrust investigations in the US and EU targeting the App Store.'], outlook: 'Management expressed confidence in the product pipeline, highlighting upcoming innovations in wearables and augmented reality. Focus remains on expanding the services portfolio and returning capital to shareholders through buybacks and dividends.' },
    'MSFT': { company: 'Microsoft Corporation (MSFT)', date: 'October 26, 2025', sentiment: 'Positive', sentimentScore: '8.5/10', summary: 'Microsoft maintains its position as a cloud computing leader with Azure showing consistent growth. The company\'s strategic investments in AI through partnerships and acquisitions position it well for future revenue streams. Integration of AI across product lines, particularly Office 350 and GitHub, demonstrates strong execution.', risks: ['Increasing competition in cloud services from AWS and Google Cloud.', 'Regulatory scrutiny over AI partnerships and acquisitions.', 'Enterprise spending slowdown amid economic uncertainty.'], outlook: 'Leadership emphasized continued investment in AI infrastructure and cloud capabilities. Strong commercial bookings and expanding margins in the cloud segment indicate sustained momentum. Focus on cybersecurity solutions presents additional growth vectors.' },
    'GOOGL': { company: 'Alphabet Inc. (GOOGL)', date: 'October 26, 2025', sentiment: 'Neutral', sentimentScore: '7.1/10', summary: 'Alphabet faces a transitional period as AI-powered search transforms its core business model. While YouTube and Cloud divisions show promise, advertising revenue growth has moderated. The company\'s significant investments in AI and quantum computing represent both opportunities and execution risks.', risks: ['Existential threat to search business from AI chatbots and competitors.', 'Antitrust actions threatening core business structure and practices.', 'Increased regulatory costs and content moderation challenges on YouTube.'], outlook: 'Management outlined aggressive AI integration plans across all products. Despite near-term headwinds, the company\'s research capabilities and data advantages provide long-term competitive positioning. Cost discipline initiatives aim to improve operational efficiency.' },
    'JPM': { company: 'JPMorgan Chase & Co. (JPM)', date: 'October 26, 2025', sentiment: 'Positive', sentimentScore: '7.8/10', summary: 'JPMorgan continues to demonstrate resilient performance across its diversified business segments. Net interest income benefits from higher rates, while investment banking shows signs of recovery. The bank\'s technology investments and risk management framework position it favorably relative to peers.', risks: ['Potential credit deterioration as economic conditions soften.', 'Regulatory capital requirements limiting shareholder returns.', 'Geopolitical instability affecting global markets and trading revenues.'], outlook: 'CEO emphasized the bank\'s fortress balance sheet and disciplined approach to capital deployment. Strategic focus on digital transformation and payments infrastructure expected to drive long-term growth. Cautious stance on near-term economic outlook reflected in conservative guidance.' },
    'PG': { company: 'Procter & Gamble Co. (PG)', date: 'October 26, 2025', sentiment: 'Neutral', sentimentScore: '6.5/10', summary: 'PG summary...', risks: [], outlook: 'PG outlook...' },
    'JNJ': { company: 'Johnson & Johnson (JNJ)', date: 'October 26, 2025', sentiment: 'Positive', sentimentScore: '7.2/10', summary: 'JNJ summary...', risks: [], outlook: 'JNJ outlook...' },
    'XOM': { company: 'Exxon Mobil Corporation (XOM)', date: 'October 26, 2025', sentiment: 'Neutral', sentimentScore: '6.8/10', summary: 'XOM summary...', risks: [], outlook: 'XOM outlook...' },
};

const complianceData = {
    activities: ['Retail Onboarding', 'Corporate Lending', 'Trading Operations', 'Wealth Management', 'Cross-Border Payments'],
    regulations: ['AML/KYC', 'Basel III', 'MiFID II', 'ESG Reporting'],
    heatmap: [['H', 'M', 'L', 'M'], ['M', 'H', 'M', 'L'], ['H', 'H', 'H', 'M'], ['M', 'L', 'M', 'H'], ['H', 'M', 'L', 'L']]
};
// --- Types ---
type ChatMessage = { sender: 'user' | 'bot'; text: string };

type LivePortfolioData = {
    optimal_weights: Record<string, number>;
    kpis: { return: number; volatility: number; sharpe_ratio: number };
};

type BriefData = {
    ticker: string;
    summary: string;
    status: string;
};

function App(): JSX.Element {
  const [activeView, setActiveView] = useState('portfolio');
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [isLoadingBrief, setIsLoadingBrief] = useState(false);
  const [isLoadingPortfolio, setIsLoadingPortfolio] = useState(false);
  const [livePortfolioData, setLivePortfolioData] = useState<LivePortfolioData | null>(null);
  const [briefData, setBriefData] = useState<BriefData | null>(null);
  
  // --- NEW: State for user-defined portfolio ---
  const [tickerInput, setTickerInput] = useState('AAPL, MSFT, JPM, PG, JNJ, XOM');

  // --- NEW: State for the Analyst Chat ---
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  // Fetch brief from API when stock changes
  const handleStockChange = async (ticker: string) => {
    setSelectedStock(ticker);
    setChatMessages([]); // Clear chat history when stock changes
    setIsLoadingBrief(true);
    setBriefData(null);

    try {
      const response = await axios.get(`http://127.0.0.1:8000/api/generate-brief/${ticker}`);
      setBriefData(response.data);
    } catch (error) {
      console.error(`Error fetching brief for ${ticker}:`, error);
      setBriefData({
        ticker: ticker,
        summary: "Failed to load brief. Please ensure the backend server is running and has the necessary API keys configured.",
        status: "error"
      });
    } finally {
      setIsLoadingBrief(false);
    }
  };

  const runOptimization = async () => {
    setIsLoadingPortfolio(true);
    setLivePortfolioData(null);
    try {
      // --- NEW: Process user input into a clean array of tickers ---
      const tickers = tickerInput.split(',')
        .map(t => t.trim())
        .filter(t => t)
        .map(t => t.toUpperCase());

      if (tickers.length < 2) {
        alert("Please enter at least two tickers for optimization.");
        setIsLoadingPortfolio(false);
        return;
      }

      const response = await axios.post('http://127.0.0.1:8000/api/optimize-portfolio', { tickers });
      setLivePortfolioData(response.data);
    } catch (error) {
      console.error("Error running optimization:", error);
      alert("Failed to optimize portfolio. Please check your tickers and ensure the backend server is running.");
    } finally {
      setIsLoadingPortfolio(false);
    }
  };

  // --- NEW: Function to handle Analyst Chat submission ---
  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || isChatLoading || !briefData) return;

    const userMessage: ChatMessage = { sender: 'user', text: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setIsChatLoading(true);

    try {
        const response = await axios.post('http://127.0.0.1:8000/api/generate/chat-response', {
            ticker: selectedStock,
            prompt: chatInput,
            brief_summary: briefData.summary
        });
        const botMessage: ChatMessage = { sender: 'bot', text: String(response.data.response) };
        setChatMessages(prev => [...prev, botMessage]);
    } catch (error) {
        console.error("Error fetching chat response:", error);
        const errorMessage: ChatMessage = { sender: 'bot', text: "Sorry, I couldn't get a response. Please try again." };
        setChatMessages(prev => [...prev, errorMessage]);
    } finally {
        setIsChatLoading(false);
    }
  };

  // Helper functions
  const getRiskLevelColor = (level: string) => level === 'H' ? 'bg-red-500' : level === 'M' ? 'bg-amber-500' : 'bg-green-500';

  // Dynamic Chart Data
  const chartData = {
    labels: livePortfolioData ? Object.keys(livePortfolioData.optimal_weights) : [],
    datasets: [{
      data: livePortfolioData ? Object.values(livePortfolioData.optimal_weights).map(w => (w as number) * 100) : [],
      backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#6b7280', '#ec4899', '#d946ef'],
      borderColor: '#111827', // Match dark sidebar
      borderWidth: 2
    }]
  };
  const chartOptions = {
    responsive: true, maintainAspectRatio: true,
    plugins: {
      legend: { position: 'bottom' as const, labels: { padding: 15, font: { size: 12 }, color: '#4b5563' } },
      tooltip: { callbacks: { label: (context: any) => `${context.label}: ${context.parsed.toFixed(2)}%` } }
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-900 text-white flex flex-col flex-shrink-0">
        <div className="p-6 border-b border-gray-700 flex items-center space-x-3">
          <BotIcon className="w-8 h-8 text-blue-400" />
          <h1 className="text-xl font-bold">Analyst Co-Pilot</h1>
        </div>
        <nav className="flex-1 p-4 space-y-2">
            {['briefs', 'portfolio', 'compliance'].map(view => (
                <button
                    key={view}
                    onClick={() => setActiveView(view)}
                    className={`w-full text-left flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${activeView === view ? 'bg-blue-600 text-white' : 'hover:bg-gray-800 text-gray-300'}`}
                >
                    <span className="font-medium capitalize">{view === 'briefs' ? 'Investment Briefs' : view === 'portfolio' ? 'MPT Portfolio' : 'Compliance Heatmap'}</span>
                </button>
            ))}
        </nav>
        <div className="p-6 border-t border-gray-700">
            <p className="text-xs text-gray-400 leading-relaxed">
                This platform integrates quantitative analysis with GenAI insights, inspired by the McKinsey framework for finance.
            </p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-8">
        {activeView === 'portfolio' && (
          <div className="max-w-5xl mx-auto">
            <div className="mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">Modern Portfolio Theory (MPT) Analysis</h2>
              <p className="text-gray-600">Enter a custom portfolio of tickers to find the optimal allocation for the maximum Sharpe Ratio.</p>
            </div>
            
            {/* --- NEW: User Input for Portfolio --- */}
            <div className="mb-6 bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <label htmlFor="ticker-input" className="block text-sm font-medium text-gray-700 mb-2">Enter Stock Tickers (comma-separated)</label>
                <div className="flex items-center space-x-4">
                    <input
                        id="ticker-input"
                        type="text"
                        value={tickerInput}
                        onChange={(e) => setTickerInput(e.target.value)}
                        placeholder="e.g., AAPL, GOOG, TSLA, JPM"
                        className="flex-grow px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                        onClick={runOptimization}
                        disabled={isLoadingPortfolio}
                        className="px-6 py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
                    >
                        {isLoadingPortfolio ? (
                            <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        ) : "Optimize"}
                    </button>
                </div>
            </div>
            
            {isLoadingPortfolio && (
                 <div className="text-center py-20 bg-white rounded-xl shadow-sm border border-gray-200">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600 font-medium">Fetching historical data and running optimization...</p>
                </div>
            )}

            {livePortfolioData && !isLoadingPortfolio && (
                <div className="space-y-8">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"><div className="text-sm font-medium text-gray-600 mb-1">Annualized Return</div><div className="text-3xl font-bold text-green-600">{(livePortfolioData.kpis.return * 100).toFixed(2)}%</div></div>
                        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"><div className="text-sm font-medium text-gray-600 mb-1">Annual Volatility</div><div className="text-3xl font-bold text-amber-600">{(livePortfolioData.kpis.volatility * 100).toFixed(2)}%</div></div>
                        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"><div className="text-sm font-medium text-gray-600 mb-1">Sharpe Ratio</div><div className="text-3xl font-bold text-blue-600">{livePortfolioData.kpis.sharpe_ratio.toFixed(2)}</div></div>
                    </div>
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                      <h3 className="text-xl font-semibold text-gray-900 mb-6 text-center">Optimized Portfolio Allocation</h3>
                      <div className="max-w-md mx-auto"><Doughnut data={chartData} options={chartOptions} /></div>
                    </div>
                </div>
            )}
          </div>
        )}

        {activeView === 'briefs' && (
           <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
                <div className="mb-8">
                    <h2 className="text-3xl font-bold text-gray-900 mb-2">Automated Investment Briefs</h2>
                    <p className="text-gray-600">AI-generated summaries of market sentiment and key risk factors.</p>
                </div>
                <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-2">Select Company</label>
                    <select
                        value={selectedStock}
                        onChange={(e) => handleStockChange(e.target.value)}
                        className="w-full max-w-xs px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                    >
                        {Object.keys(briefsData).map(ticker => <option key={ticker} value={ticker}>{briefsData[ticker as keyof typeof briefsData].company}</option>)}
                    </select>
                </div>
                {isLoadingBrief ? (
                  <div className="text-center py-20 bg-white rounded-xl shadow-sm border border-gray-200">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600 font-medium">Fetching investment brief from SEC API...</p>
                  </div>
                ) : briefData ? (
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200">
                        <div className="p-6 border-b border-gray-200">
                            <div className="mb-2"><h3 className="text-2xl font-bold text-gray-900">{selectedStock}</h3></div>
                            <div className="text-sm text-gray-600 italic">AI-generated summary from latest 10-K filing</div>
                        </div>
                        <div className="p-6 space-y-6">
                            <div><h4 className="text-lg font-semibold text-gray-900 mb-3">Management Discussion & Analysis (MD&A)</h4><p className="text-gray-700 leading-relaxed">{briefData.summary}</p></div>
                        </div>
                    </div>
                ) : (
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                        <p className="text-gray-600">No brief available. Please check the backend is running.</p>
                    </div>
                )}
            </div>
            {/* --- NEW: Analyst Chat Component --- */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col h-[70vh] mt-24">
                <div className="p-4 border-b border-gray-200"><h3 className="font-semibold text-gray-800">Ask the Analyst about {selectedStock}</h3></div>
                <div className="flex-grow p-4 overflow-y-auto bg-gray-50 space-y-4">
                    {chatMessages.map((msg, index) => (
                        <div key={index} className={`flex items-start gap-3 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {msg.sender === 'bot' && <BotIcon className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />}
                            <div className={`max-w-xs md:max-w-md p-3 rounded-2xl ${msg.sender === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-gray-200 text-gray-800 rounded-bl-none'}`}>
                                <p className="text-sm leading-relaxed">{msg.text}</p>
                            </div>
                            {msg.sender === 'user' && <UserIcon className="w-6 h-6 text-gray-500 flex-shrink-0 mt-1" />}
                        </div>
                    ))}
                     {isChatLoading && (
                        <div className="flex items-start gap-3 justify-start">
                            <BotIcon className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
                            <div className="max-w-xs p-3 rounded-2xl bg-gray-200 text-gray-800 rounded-bl-none">
                                <div className="flex items-center space-x-2">
                                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div>
                                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse delay-75"></div>
                                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse delay-150"></div>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={chatEndRef} />
                </div>
                <form onSubmit={handleChatSubmit} className="p-4 border-t border-gray-200 bg-white">
                    <div className="flex items-center space-x-2">
                        <input
                            type="text"
                            value={chatInput}
                            onChange={(e) => setChatInput(e.target.value)}
                            placeholder="Ask a follow-up question..."
                            className="flex-grow px-4 py-2 border border-gray-300 rounded-full focus:ring-2 focus:ring-blue-500"
                            disabled={isChatLoading}
                        />
                        <button type="submit" className="p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 disabled:bg-gray-400" disabled={isChatLoading}>
                            <SendIcon className="w-5 h-5"/>
                        </button>
                    </div>
                </form>
            </div>
           </div>
        )}

        {activeView === 'compliance' && (
           <div className="max-w-6xl mx-auto">
                <div className="mb-8"><h2 className="text-3xl font-bold text-gray-900 mb-2">Regulatory Compliance Heatmap</h2><p className="text-gray-600">Risk assessment across banking activities and regulatory frameworks.</p></div>
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead className="bg-gray-50"><tr className="border-b border-gray-200">{['Activity', ...complianceData.regulations].map((h, i) => <th key={i} className={`px-6 py-4 text-sm font-semibold text-gray-900 ${i===0 ? 'text-left' : 'text-center'}`}>{h}</th>)}</tr></thead>
                            <tbody>
                                {complianceData.activities.map((activity, rowIndex) => (
                                <tr key={rowIndex} className="border-b border-gray-200 last:border-0 hover:bg-gray-50">
                                    <td className="px-6 py-4 font-medium text-gray-900">{activity}</td>
                                    {complianceData.heatmap[rowIndex].map((level, colIndex) => (
                                    <td key={colIndex} className="px-6 py-4 text-center">
                                        <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-white font-bold text-xs shadow-md ${getRiskLevelColor(level)}`}>{level}</span>
                                    </td>
                                    ))}
                                </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <div className="p-6 bg-gray-50 border-t border-gray-200">
                        <div className="flex items-center justify-center space-x-8">
                            <div className="flex items-center space-x-2"><span className="w-4 h-4 rounded-full bg-red-500"></span><span className="text-sm font-medium text-gray-700">High Risk</span></div>
                            <div className="flex items-center space-x-2"><span className="w-4 h-4 rounded-full bg-amber-500"></span><span className="text-sm font-medium text-gray-700">Medium Risk</span></div>
                            <div className="flex items-center space-x-2"><span className="w-4 h-4 rounded-full bg-green-500"></span><span className="text-sm font-medium text-gray-700">Low Risk</span></div>
                        </div>
                    </div>
                </div>
           </div>
        )}
      </main>
    </div>
  );
}

export default App;

