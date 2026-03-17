<script lang="ts">
	import { run } from 'svelte/legacy';

	import { onMount, onDestroy, createEventDispatcher } from "svelte";
	import type {
		IChartApi,
		ISeriesApi,
		CandlestickData,
		HistogramData,
		Time,
		DeepPartial,
		ChartOptions,
		CandlestickSeriesOptions,
		HistogramSeriesOptions,
	} from "lightweight-charts";
	import { createChart, ColorType, CrosshairMode } from "lightweight-charts";
	import { wsClient } from "$lib/ws-client";

	
	
	interface Props {
		// Props
		symbol?: string;
		timeframe?: string;
		data?: CandlestickData<Time>[];
		volumeData?: HistogramData<Time>[];
		trades?: TradeMarker[];
		regimes?: RegimeOverlay[];
		autoSize?: boolean;
		showVolume?: boolean;
		showTrades?: boolean;
		showRegimes?: boolean;
		wsEnabled?: boolean;
		// WebSocket URL prop - defaults to relative /ws/chart endpoint
		wsUrl?: string;
	}

	let {
		symbol = "EURUSD",
		timeframe = "H1",
		data = $bindable([]),
		volumeData = [],
		trades = $bindable([]),
		regimes = $bindable([]),
		autoSize = true,
		showVolume = true,
		showTrades = true,
		showRegimes = true,
		wsEnabled = true,
		wsUrl = ""
	}: Props = $props();

	// Internal state
	let ws: WebSocket | null = null;
	let isConnected: boolean = false;
	let reconnectAttempts: number = 0;
	const MAX_RECONNECT_ATTEMPTS = 5;
	const RECONNECT_DELAY = 3000;

	// Chart container reference
	let chartContainer: HTMLDivElement = $state();
	let chart: IChartApi | null = $state(null);
	let candlestickSeries: ISeriesApi<"Candlestick"> | null = $state(null);
	let volumeSeries: ISeriesApi<"Histogram"> | null = $state(null);
	let tradeMarkers: any[] = [];

	// Dispatch events
	const dispatch = createEventDispatcher();

	// Types
	interface TradeMarker {
		time: Time;
		price: number;
		type: "buy" | "sell";
		quantity?: number;
	}

	interface RegimeOverlay {
		startTime: Time;
		endTime: Time;
		regime: "TRENDING" | "RANGING" | "VOLATILE" | "CHOPPY";
	}

	// Color scheme - Midnight Finance
	const colors = {
		background: "#0f172a", // Slate 900
		text: "#94a3b8", // Slate 400
		grid: "#1e293b", // Slate 800
		upColor: "#10b981", // Emerald 500
		downColor: "#f43f5e", // Rose 500
		volumeUp: "rgba(16, 185, 129, 0.4)",
		volumeDown: "rgba(244, 63, 94, 0.4)",
		borderUp: "#10b981",
		borderDown: "#f43f5e",
		wickUp: "#10b981",
		wickDown: "#f43f5e",
		regimeColors: {
			TRENDING: "rgba(59, 130, 246, 0.1)",
			RANGING: "rgba(245, 158, 11, 0.1)",
			VOLATILE: "rgba(244, 63, 94, 0.1)",
			CHOPPY: "rgba(139, 92, 246, 0.1)",
		},
	};

	// Chart options
	const chartOptions: DeepPartial<ChartOptions> = {
		layout: {
			background: { type: ColorType.Solid, color: colors.background },
			textColor: colors.text,
			fontFamily: "'Inter', system-ui, sans-serif",
		},
		grid: {
			vertLines: { color: colors.grid },
			horzLines: { color: colors.grid },
		},
		width: 800,
		height: 500,
		crosshair: {
			mode: CrosshairMode.Normal,
			vertLine: {
				color: "#6b7280",
				width: 1,
				style: 2,
				labelBackgroundColor: "#374151",
			},
			horzLine: {
				color: "#6b7280",
				width: 1,
				style: 2,
				labelBackgroundColor: "#374151",
			},
		},
		rightPriceScale: {
			borderColor: colors.grid,
			scaleMargins: {
				top: 0.1,
				bottom: 0.2,
			},
		},
		timeScale: {
			borderColor: colors.grid,
			timeVisible: true,
			secondsVisible: false,
		},
	};

	// WebSocket connection
	function connectWebSocket() {
		if (!wsEnabled || typeof WebSocket === "undefined") return;

		const wsEndpoint = wsUrl || `/ws/chart/${symbol}/${timeframe}`;
		const wsProtocol =
			window.location.protocol === "https:" ? "wss:" : "ws:";
		const wsHost = wsUrl
			? wsUrl
			: `${wsProtocol}//${window.location.host}${wsEndpoint}`;

		try {
			ws = new WebSocket(wsHost);

			ws.onopen = () => {
				console.log("Chart WebSocket connected");
				isConnected = true;
				reconnectAttempts = 0;
				dispatch("ws-connected", { symbol, timeframe });
			};

			ws.onmessage = (event) => {
				try {
					const message = JSON.parse(event.data);
					handleWsMessage(message);
				} catch (e) {
					console.error("Failed to parse WebSocket message:", e);
				}
			};

			ws.onclose = () => {
				console.log("Chart WebSocket disconnected");
				isConnected = false;
				dispatch("ws-disconnected", { symbol, timeframe });

				// Attempt reconnect
				if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
					setTimeout(() => {
						reconnectAttempts++;
						connectWebSocket();
					}, RECONNECT_DELAY);
				}
			};

			ws.onerror = (error) => {
				console.error("Chart WebSocket error:", error);
				dispatch("ws-error", { error });
			};
		} catch (e) {
			console.error("Failed to create WebSocket:", e);
		}
	}

	// Handle WebSocket messages
	function handleWsMessage(message: any) {
		switch (message.type) {
			case "subscribed":
				console.log("Subscribed to chart:", message);
				break;

			case "historical_bars":
				if (message.data && message.data.length > 0) {
					const bars = message.data.map((bar: any) => ({
						time: bar.time as Time,
						open: bar.open,
						high: bar.high,
						low: bar.low,
						close: bar.close,
					}));
					data = bars;
					if (candlestickSeries) {
						candlestickSeries.setData(bars);
					}
				}
				break;

			case "ohlcv":
				if (message.data) {
					const bar = message.data;
					const candleData: CandlestickData<Time> = {
						time: bar.time as Time,
						open: bar.open,
						high: bar.high,
						low: bar.low,
						close: bar.close,
					};
					addBar(
						candleData,
						bar.volume
							? {
									time: bar.time as Time,
									value: bar.volume,
									color:
										bar.close >= bar.open
											? colors.volumeUp
											: colors.volumeDown,
								}
							: undefined,
					);
					dispatch("bar-update", candleData);
				}
				break;

			case "regime":
				if (message.data) {
					regimes = [
						...regimes,
						{
							startTime: message.data.time as Time,
							endTime: message.data.endTime as Time,
							regime: message.data.regime,
						},
					];
					dispatch("regime-change", message.data);
				}
				break;

			case "trade":
				if (message.data) {
					const trade: TradeMarker = {
						time: message.data.time as Time,
						price: message.data.price,
						type: message.data.direction === "buy" ? "buy" : "sell",
						quantity: message.data.volume,
					};
					addTrade(trade);
					dispatch("trade", trade);
				}
				break;
		}
	}

	// Disconnect WebSocket
	function disconnectWebSocket() {
		if (ws) {
			ws.close();
			ws = null;
		}
	}

	// Initialize chart
	onMount(() => {
		if (!chartContainer) return;

		// Connect to WebSocket if enabled
		if (wsEnabled) {
			connectWebSocket();
		}

		// Create chart
		chart = createChart(chartContainer, {
			...chartOptions,
			width: chartContainer.clientWidth || 800,
		});

		// Add candlestick series
		candlestickSeries = chart.addCandlestickSeries({
			upColor: colors.upColor,
			downColor: colors.downColor,
			borderUpColor: colors.borderUp,
			borderDownColor: colors.borderDown,
			wickUpColor: colors.wickUp,
			wickDownColor: colors.wickDown,
			borderVisible: false,
			wickVisible: true,
		} as CandlestickSeriesOptions);

		// Add volume series
		if (showVolume) {
			volumeSeries = chart.addHistogramSeries({
				color: colors.volumeUp,
				priceFormat: {
					type: "volume",
				},
				priceScaleId: "",
				scaleMargins: {
					top: 0.85,
					bottom: 0,
				},
				base: 0,
			} as HistogramSeriesOptions);
		}

		// Set initial data
		if (data.length > 0) {
			candlestickSeries.setData(data);
		}

		if (volumeData.length > 0 && volumeSeries) {
			volumeSeries.setData(volumeData);
		}

		// Add trade markers
		if (showTrades && trades.length > 0) {
			updateTradeMarkers();
		}

		// Add regime overlays
		if (showRegimes && regimes.length > 0) {
			updateRegimeOverlays();
		}

		// Fit content
		chart.timeScale().fitContent();

		// Handle resize
		if (autoSize) {
			const resizeObserver = new ResizeObserver((entries) => {
				if (chart && entries.length > 0) {
					const { width } = entries[0].contentRect;
					chart.applyOptions({ width });
				}
			});
			resizeObserver.observe(chartContainer);

			return () => {
				resizeObserver.disconnect();
			};
		}

		// Subscribe to crosshair move
		chart.subscribeCrosshairMove((param) => {
			if (!param.time || !param.seriesData) return;

			if (!candlestickSeries) return;
			const candleData = param.seriesData.get(candlestickSeries);
			if (candleData) {
				dispatch("crosshair-move", {
					time: param.time,
					price: (candleData as any).close,
					data: candleData,
				});
			}
		});

		// Subscribe to visible range change
		chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
			dispatch("visible-range-change", { range });
		});
	});

	onDestroy(() => {
		// Disconnect WebSocket
		disconnectWebSocket();

		if (chart) {
			chart.remove();
			chart = null;
		}
	});

	// Update data
	run(() => {
		if (chart && candlestickSeries && data.length > 0) {
			candlestickSeries.setData(data);
		}
	});

	run(() => {
		if (chart && volumeSeries && volumeData.length > 0) {
			volumeSeries.setData(volumeData);
		}
	});

	// React to symbol/timeframe changes - reconnect WebSocket
	run(() => {
		if (wsEnabled && (symbol || timeframe)) {
			disconnectWebSocket();
			if (chartContainer) {
				connectWebSocket();
			}
		}
	});

	// Update trade markers
	function updateTradeMarkers() {
		if (!candlestickSeries || !showTrades) return;

		const markers = trades.map((trade) => ({
			time: trade.time,
			position:
				trade.type === "buy"
					? ("belowBar" as const)
					: ("aboveBar" as const),
			color: trade.type === "buy" ? colors.upColor : colors.downColor,
			shape:
				trade.type === "buy"
					? ("arrowUp" as const)
					: ("arrowDown" as const),
			text: trade.type === "buy" ? "▲ BUY" : "▼ SELL",
		}));

		candlestickSeries.setMarkers(markers);
	}

	// Update regime overlays
	function updateRegimeOverlays() {
		if (!chart || !showRegimes) return;

		regimes.forEach((regime) => {
			// Create price lines for regime boundaries
			const color =
				colors.regimeColors[regime.regime] ||
				colors.regimeColors.RANGING;

			// Note: Lightweight Charts doesn't have built-in background regions,
			// but we can use price lines or create custom overlays
		});
	}

	// Add real-time bar update
	export function updateBar(bar: CandlestickData<Time>) {
		if (!candlestickSeries) return;
		candlestickSeries.update(bar);
	}

	// Add new bar
	export function addBar(
		bar: CandlestickData<Time>,
		volume?: HistogramData<Time>,
	) {
		if (!candlestickSeries) return;
		candlestickSeries.update(bar);

		if (volume && volumeSeries) {
			volumeSeries.update(volume);
		}
	}

	// Add trade marker
	export function addTrade(trade: TradeMarker) {
		trades = [...trades, trade];
		updateTradeMarkers();
	}

	// Set visible range
	export function setVisibleRange(from: Time, to: Time) {
		if (!chart) return;
		chart.timeScale().setVisibleRange({ from, to });
	}

	// Fit content
	export function fitContent() {
		if (!chart) return;
		chart.timeScale().fitContent();
	}

	// Scroll to real-time
	export function scrollToRealTime() {
		if (!chart) return;
		chart.timeScale().scrollToRealTime();
	}

	// Get current price
	export function getCurrentPrice(): number | null {
		if (!candlestickSeries || !data.length) return null;
		return data[data.length - 1].close;
	}

	// Apply options
	export function applyOptions(options: DeepPartial<ChartOptions>) {
		if (!chart) return;
		chart.applyOptions(options);
	}

	// Take screenshot
	export function takeScreenshot(): string | null {
		if (!chart) return null;
		return chart.takeScreenshot().toDataURL();
	}
</script>

<div class="tradingview-chart-wrapper">
	<div class="chart-header">
		<div class="symbol-info">
			<span class="symbol">{symbol}</span>
			<span class="timeframe">{timeframe}</span>
		</div>
		<div class="chart-controls">
			<button
				class="control-btn"
				onclick={fitContent}
				title="Fit Content"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					width="16"
					height="16"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
				>
					<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
				</svg>
			</button>
			<button
				class="control-btn"
				onclick={scrollToRealTime}
				title="Scroll to Real-time"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					width="16"
					height="16"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
				>
					<circle cx="12" cy="12" r="10" />
					<polyline points="12 6 12 12 16 14" />
				</svg>
			</button>
		</div>
	</div>
	<div bind:this={chartContainer} class="chart-container"></div>
</div>

<style>
	.tradingview-chart-wrapper {
		display: flex;
		flex-direction: column;
		width: 100%;
		height: 100%;
		background-color: #0a0a0a;
		border-radius: 8px;
		overflow: hidden;
	}

	.chart-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 12px 16px;
		background-color: #111827;
		border-bottom: 1px solid #1f2937;
	}

	.symbol-info {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.symbol {
		font-size: 18px;
		font-weight: 600;
		color: #f3f4f6;
	}

	.timeframe {
		font-size: 14px;
		color: #6b7280;
		padding: 2px 8px;
		background-color: #1f2937;
		border-radius: 4px;
	}

	.chart-controls {
		display: flex;
		gap: 8px;
	}

	.control-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 32px;
		height: 32px;
		background-color: #1f2937;
		border: none;
		border-radius: 6px;
		color: #9ca3af;
		cursor: pointer;
		transition: all 0.2s ease;
	}

	.control-btn:hover {
		background-color: #374151;
		color: #f3f4f6;
	}

	.chart-container {
		flex: 1;
		min-height: 400px;
	}
</style>
