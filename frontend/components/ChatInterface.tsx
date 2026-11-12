"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Box,
  Flex,
  Text,
  VStack,
  HStack,
  Badge,
  Spinner,
  Textarea,
  IconButton,
  Code,
  Accordion,
  Button,
} from "@chakra-ui/react";
import ReactMarkdown from "react-markdown";
import { InlineGraph } from "./ContextGraphView";
import {
  streamChatMessage,
  getGraphData,
  type ChatMessage,
  type StreamEvent,
  type Decision,
  type GraphData,
  type AgentContext,
  type FastAnswer,
  type JevVerdict,
  type PolicyGuard,
  type DecisionMode,
  type FinalRecommendation,
} from "@/lib/api";

interface ChatInterfaceProps {
  conversationHistory: ChatMessage[];
  onConversationUpdate: (messages: ChatMessage[]) => void;
  onDecisionSelect: (decision: Decision) => void;
  onGraphUpdate: (data: GraphData) => void;
}

interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  output?: unknown;
}

interface MessageWithGraph extends ChatMessage {
  graphData?: GraphData;
  toolCalls?: ToolCall[];
  agentContext?: AgentContext;
  fastAnswer?: FastAnswer;
  additionalToolsUsed?: string[];
  finalRecommendation?: FinalRecommendation | null;
}

// Helper function to extract entity IDs from tool results
function extractEntityIds(
  toolName: string,
  input: Record<string, unknown>,
  output: unknown,
): string[] {
  const ids: string[] = [];

  // Extract from input parameters
  if (input.customer_id) ids.push(String(input.customer_id));
  if (input.account_id) ids.push(String(input.account_id));
  if (input.decision_id) ids.push(String(input.decision_id));

  // Extract from output based on tool type
  if (output && typeof output === "object") {
    const result = output as Record<string, unknown>;

    // Handle customers array
    if (Array.isArray(result.customers)) {
      result.customers.slice(0, 3).forEach((c: Record<string, unknown>) => {
        if (c.id) ids.push(String(c.id));
      });
    }

    // Handle decisions array
    if (Array.isArray(result.decisions)) {
      result.decisions.slice(0, 3).forEach((d: Record<string, unknown>) => {
        if (d.id) ids.push(String(d.id));
      });
    }

    // Handle similar_decisions array
    if (Array.isArray(result.similar_decisions)) {
      result.similar_decisions
        .slice(0, 3)
        .forEach((d: Record<string, unknown>) => {
          if (d.id) ids.push(String(d.id));
        });
    }

    // Handle precedents array
    if (Array.isArray(result.precedents)) {
      result.precedents.slice(0, 3).forEach((p: Record<string, unknown>) => {
        if (p.id) ids.push(String(p.id));
      });
    }

    // Handle causal_chain
    if (result.causal_chain && typeof result.causal_chain === "object") {
      const chain = result.causal_chain as Record<string, unknown>;
      if (chain.decision_id) ids.push(String(chain.decision_id));
    }
  }

  return Array.from(new Set(ids)); // Remove duplicates
}

export function ChatInterface({
  conversationHistory,
  onConversationUpdate,
  onDecisionSelect,
  onGraphUpdate,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<MessageWithGraph[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  // How the fast path decides: code guards over per-item Jev judgments, or one Jev Choice.
  const [decisionMode, setDecisionMode] = useState<DecisionMode>("decomposed");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom only when user sends a new message (not during streaming updates)
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [input]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: MessageWithGraph = {
      role: "user",
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    scrollToBottom();

    // Create a placeholder for the streaming assistant message
    const assistantMessageIndex = messages.length + 1;
    const assistantMessage: MessageWithGraph = {
      role: "assistant",
      content: "",
      toolCalls: [],
    };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      const toolCalls: ToolCall[] = [];
      let fullContent = "";
      let graphData: GraphData | undefined;
      let agentContext: AgentContext | undefined;
      let fastAnswer: FastAnswer | undefined;
      let additionalToolsUsed: string[] | undefined;
      let finalRecommendation: FinalRecommendation | null | undefined;

      for await (const event of streamChatMessage(
        userMessage.content,
        messages.map((m) => ({ role: m.role, content: m.content })),
        decisionMode,
      )) {
        switch (event.type) {
          case "agent_context":
            agentContext = event.context;
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                agentContext,
              };
              return updated;
            });
            break;

          case "fast_answer":
            fastAnswer = event.fastAnswer;
            if (fastAnswer.applicable && fastAnswer.graph_data?.nodes?.length) {
              graphData = fastAnswer.graph_data;
              onGraphUpdate(graphData);
            }
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                fastAnswer,
                graphData,
              };
              return updated;
            });
            break;

          case "text":
            fullContent += event.content + "\n\n";
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                content: fullContent,
              };
              return updated;
            });
            break;

          case "tool_use":
            toolCalls.push({ name: event.name, input: event.input });
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                toolCalls: [...toolCalls],
              };
              return updated;
            });
            break;

          case "tool_result":
            // Update the matching tool call with its output
            console.log("tool_result event:", event);
            console.log("Looking for tool with name:", event.name);
            console.log(
              "Current toolCalls:",
              JSON.stringify(toolCalls, null, 2),
            );
            const toolIndex = toolCalls.findIndex(
              (t) => t.name === event.name && t.output === undefined,
            );
            console.log("Found toolIndex:", toolIndex);
            if (toolIndex !== -1) {
              const currentTool = toolCalls[toolIndex];
              toolCalls[toolIndex].output = event.output;
              console.log("Updated toolCall output:", toolCalls[toolIndex]);

              // Check if this tool returned graph data directly
              let foundGraphData = false;
              if (event.output && typeof event.output === "object") {
                const result = event.output as Record<string, unknown>;
                // Check for graph_data field (new format from agent tools)
                if (
                  result.graph_data &&
                  typeof result.graph_data === "object"
                ) {
                  const gd = result.graph_data as Record<string, unknown>;
                  if (gd.nodes && gd.relationships) {
                    graphData = gd as unknown as GraphData;
                    onGraphUpdate(graphData);
                    foundGraphData = true;
                  }
                }
                // Also check for direct nodes/relationships (legacy format)
                else if (result.nodes && result.relationships) {
                  graphData = result as unknown as GraphData;
                  onGraphUpdate(graphData);
                  foundGraphData = true;
                }
              }

              // If no graph data was returned, try to fetch it based on entity IDs
              if (!foundGraphData && event.output) {
                const entityIds = extractEntityIds(
                  currentTool.name,
                  currentTool.input,
                  event.output,
                );
                if (entityIds.length > 0) {
                  // Fetch graph data for the first entity ID found
                  getGraphData(entityIds[0], 2)
                    .then((fetchedGraphData) => {
                      if (fetchedGraphData.nodes.length > 0) {
                        graphData = fetchedGraphData;
                        onGraphUpdate(fetchedGraphData);
                        // Update message with graph data
                        setMessages((prev) => {
                          const updated = [...prev];
                          if (updated[assistantMessageIndex]) {
                            updated[assistantMessageIndex] = {
                              ...updated[assistantMessageIndex],
                              graphData: fetchedGraphData,
                            };
                          }
                          return updated;
                        });
                      }
                    })
                    .catch((err) => {
                      console.error("Failed to fetch graph data:", err);
                    });
                }
              }

              setMessages((prev) => {
                const updated = [...prev];
                updated[assistantMessageIndex] = {
                  ...updated[assistantMessageIndex],
                  toolCalls: [...toolCalls],
                  graphData,
                };
                return updated;
              });
            }
            break;

          case "done":
            // Final update with complete data
            // Use local toolCalls array which has outputs populated from tool_result events
            additionalToolsUsed = event.additional_tools_used ?? [];
            finalRecommendation = event.final_recommendation ?? null;
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                content: fullContent,
                toolCalls: toolCalls,
                graphData,
                agentContext,
                fastAnswer,
                additionalToolsUsed,
                finalRecommendation,
              };
              return updated;
            });
            break;

          case "error":
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                content: `Error: ${event.error}`,
              };
              return updated;
            });
            break;
        }
      }

      // Update conversation history
      const finalMessage: MessageWithGraph = {
        role: "assistant",
        content: fullContent,
        toolCalls,
        graphData,
        agentContext,
        fastAnswer,
        additionalToolsUsed,
        finalRecommendation,
      };
      onConversationUpdate([...messages, userMessage, finalMessage]);
    } catch (error) {
      console.error("Failed to send message:", error);
      setMessages((prev) => {
        const updated = [...prev];
        updated[assistantMessageIndex] = {
          role: "assistant",
          content:
            "Sorry, I encountered an error processing your request. Please try again.",
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  }, [
    decisionMode,
    input,
    isLoading,
    messages,
    onConversationUpdate,
    onGraphUpdate,
    scrollToBottom,
  ]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Flex direction="column" h="100%">
      {/* Messages Area */}
      <Box flex={1} overflow="auto" p={4} pb={{ base: "100px", md: 4 }}>
        <VStack gap={4} align="stretch">
          {/* Welcome message */}
          {messages.length === 0 && (
            <Box
              bg="brand.50"
              _dark={{ bg: "brand.900", borderColor: "brand.700" }}
              p={4}
              borderRadius="lg"
              borderWidth="1px"
              borderColor="brand.200"
            >
              <Text fontWeight="medium" mb={2}>
                Welcome to Context Graph Demo
              </Text>
              <Text
                fontSize="sm"
                color="gray.600"
                _dark={{ color: "gray.400" }}
              >
                I can help you search for customers, analyze decisions, find
                similar precedents, and trace causal relationships. Try asking:
              </Text>
              <VStack align="start" mt={3} gap={1}>
                <SuggestionChip
                  text="Should we approve a credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase."
                  onClick={() =>
                    setInput(
                      "Should we approve a credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase.",
                    )
                  }
                />
                <SuggestionChip
                  text="Search for customer John Walsh"
                  onClick={() => setInput("Search for customer John Walsh")}
                />
                <SuggestionChip
                  text="A customer wants to make a $15,000 wire transfer. What policies apply and are there similar past decisions?"
                  onClick={() =>
                    setInput(
                      "A customer wants to make a $15,000 wire transfer. What policies apply and are there similar past decisions?",
                    )
                  }
                />
                <SuggestionChip
                  text="We need to override the trading limit for Katherine Miller. Find precedents for similar exceptions."
                  onClick={() =>
                    setInput(
                      "We need to override the trading limit for Katherine Miller. Find precedents for similar exceptions.",
                    )
                  }
                />
              </VStack>
            </Box>
          )}

          {/* Chat messages */}
          {messages.map((message, idx) => (
            <Box key={idx} mb={message.role === "assistant" ? 4 : 0}>
              <ChatMessageBubble
                message={message}
                isStreaming={
                  isLoading &&
                  idx === messages.length - 1 &&
                  message.role === "assistant"
                }
                onDecisionClick={onDecisionSelect}
                onNodeClick={(nodeId, labels) => {
                  console.log("Node clicked in chat:", nodeId, labels);
                }}
              />
            </Box>
          ))}

          {/* Loading indicator - only show when waiting for first response */}
          {isLoading &&
            messages.length > 0 &&
            messages[messages.length - 1].role === "user" && (
              <Flex align="center" gap={2} p={3}>
                <Spinner size="sm" color="brand.500" />
                <Text fontSize="sm" color="gray.500">
                  Thinking...
                </Text>
              </Flex>
            )}

          <div ref={messagesEndRef} />
        </VStack>
      </Box>

      {/* Input Area */}
      <Box
        p={{ base: 3, md: 4 }}
        borderTopWidth="1px"
        borderColor="border.default"
        position={{ base: "fixed", md: "relative" }}
        bottom={{ base: 0, md: "auto" }}
        left={{ base: 0, md: "auto" }}
        right={{ base: 0, md: "auto" }}
        bg="bg.surface"
        zIndex={10}
      >
        <HStack gap={2} mb={2} fontSize="xs" flexWrap="wrap">
          <Text color="gray.600" whiteSpace="nowrap">
            Decide with:
          </Text>
          {(
            [
              ["agent", "1 · Claude + tools"],
              ["monolithic", "2 · 10 queries + one Jev choice"],
              ["decomposed", "3 · 10 queries + Jev judgments + code"],
            ] as [DecisionMode, string][]
          ).map(([mode, label]) => (
            <Button
              key={mode}
              size="2xs"
              variant={decisionMode === mode ? "solid" : "outline"}
              colorPalette={decisionMode === mode ? "purple" : "gray"}
              onClick={() => setDecisionMode(mode)}
              aria-pressed={decisionMode === mode}
            >
              {label}
            </Button>
          ))}
        </HStack>
        <Flex gap={2}>
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about customers, decisions, or policies..."
            rows={1}
            resize="none"
            flex={1}
          />
          <IconButton
            aria-label="Send message"
            onClick={handleSend}
            colorPalette="brand"
          >
            <SendIcon />
          </IconButton>
        </Flex>
      </Box>
    </Flex>
  );
}

// Helper to format JSON for display
function formatJSON(obj: unknown): string {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

// Tool call display component
function ToolCallDisplay({
  toolCall,
  index,
}: {
  toolCall: ToolCall;
  index: number;
}) {
  return (
    <Box
      key={index}
      mt={2}
      borderLeft="3px solid"
      borderColor="blue.400"
      pl={3}
    >
      <HStack mb={2}>
        <Badge colorPalette="blue" fontSize="xs">
          Tool Call
        </Badge>
        <Text fontWeight="bold" fontSize="sm">
          {toolCall.name.replace("mcp__graph__", "")}
        </Text>
      </HStack>

      <Box mb={2}>
        <Text fontSize="xs" color="gray.600" fontWeight="semibold" mb={1}>
          Arguments:
        </Text>
        <Code
          display="block"
          whiteSpace="pre-wrap"
          p={2}
          borderRadius="md"
          fontSize="xs"
          bg="gray.50"
          maxH="200px"
          overflowY="auto"
        >
          {formatJSON(toolCall.input)}
        </Code>
      </Box>

      {toolCall.output !== undefined && (
        <Box>
          <Text fontSize="xs" color="gray.600" fontWeight="semibold" mb={1}>
            Output:
          </Text>
          <Code
            display="block"
            whiteSpace="pre-wrap"
            p={2}
            borderRadius="md"
            fontSize="xs"
            bg="gray.50"
            maxH="300px"
            overflowY="auto"
          >
            {formatJSON(toolCall.output)}
          </Code>
        </Box>
      )}
    </Box>
  );
}

// Fast answer card: Jev's typed verdict plus the evidence retrievals behind it
const VERDICT_COLORS: Record<string, string> = {
  yes: "green",
  no: "red",
  needs_llm_decision: "orange",
  approve: "green",
  reject: "red",
};

const FINAL_RECOMMENDATION_LINE = /^\s*\**\s*FINAL_RECOMMENDATION:.*$/gim;

function stripFinalRecommendation(content: string): string {
  return content.replace(FINAL_RECOMMENDATION_LINE, "").trim();
}

function FinalRecommendationRow({
  recommendation,
}: {
  recommendation: FinalRecommendation;
}) {
  const final = recommendation.final;
  if (!final) {
    return (
      <Text fontSize="xs" color="gray.600">
        Claude did not return a final recommendation line.
      </Text>
    );
  }
  const who = recommendation.escalated
    ? recommendation.mode === "decomposed"
      ? "Claude (escalated by a guard)"
      : "Claude (escalated by Jev)"
    : recommendation.mode === "decomposed"
      ? "code guards over Jev judgments"
      : "Jev";
  return (
    <HStack flexWrap="wrap" gap={2}>
      <Text fontSize="xs" color="gray.700" fontWeight="semibold">
        Final answer:
      </Text>
      <Badge colorPalette={VERDICT_COLORS[final] ?? "gray"} fontSize="md" px={2}>
        {final}
      </Badge>
      <Text fontSize="xs" color="gray.600">
        decided by {who}
        {recommendation.escalated && recommendation.confidence !== null
          ? ` · confidence ${(recommendation.confidence * 100).toFixed(0)}%`
          : ""}
      </Text>
      {recommendation.agrees_with_jev === true && (
        <Badge colorPalette="green" variant="subtle" fontSize="xs">
          Claude concurs
        </Badge>
      )}
      {recommendation.agrees_with_jev === false && (
        <Badge colorPalette="red" variant="subtle" fontSize="xs">
          Claude disagrees with Jev
        </Badge>
      )}
    </HStack>
  );
}

// Decomposed verdict: which guard decided, every guard in order, Jev's per-item
// judgments and the facts code computed. "Jev makes judgments, code makes decisions."
const GUARD_STATUS_COLORS: Record<string, string> = {
  fired: "red",
  passed: "green",
  not_applicable: "gray",
};
const BAND_COLORS: Record<string, string> = { yes: "orange", unsure: "yellow", no: "gray" };
const FAMILY_LABELS: Record<string, string> = {
  ticket_unresolved_concern: "unresolved concern in ticket",
  decision_open_condition: "unmet condition in prior decision",
  alert_unexplained_risk: "unexplained risk in alert",
  precedent_comparable: "comparable precedent",
};

function guardLabel(name: string): string {
  return name.replace(/^g_/, "").replace(/_/g, " ");
}

function GuardRow({ guard: g }: { guard: PolicyGuard }) {
  return (
    <HStack gap={2} fontSize="xs" align="start">
      <Badge
        colorPalette={
          g.status === "fired" ? (g.effect === "reject" ? "red" : "orange") : GUARD_STATUS_COLORS[g.status]
        }
        variant={g.status === "fired" ? "solid" : "subtle"}
        fontSize="xs"
        minW="64px"
        justifyContent="center"
      >
        {g.status === "fired" ? g.effect : "passed"}
      </Badge>
      <Badge colorPalette={g.source === "jev" ? "purple" : "blue"} variant="outline" fontSize="xs">
        {g.source}
      </Badge>
      <Text color="gray.700" flex={1}>
        <strong>{guardLabel(g.name)}</strong> · {g.reason}
        {g.evidence.length > 0 ? ` (${g.evidence.join("; ")})` : ""}
      </Text>
    </HStack>
  );
}

function DecomposedTrace({ verdict }: { verdict: JevVerdict }) {
  const guards = verdict.guards ?? [];
  const fired = guards.filter((g) => g.status === "fired");
  const passed = guards.filter((g) => g.status === "passed");
  const skipped = guards.filter((g) => g.status === "not_applicable");
  const judgments = verdict.judgments ?? [];
  const facts = verdict.facts ?? [];
  const decider = verdict.decided_by;
  return (
    <Box mb={2}>
      <Text fontSize="xs" color="gray.800" mb={2}>
        {decider?.guard ? (
          <>
            Decided by <strong>{decider.guard}</strong> ({decider.source === "jev" ? "Jev judgment" : "written rule"}):{" "}
            {decider.reason}.
          </>
        ) : (
          <>Every guard passed, so code approves.</>
        )}
      </Text>

      <VStack align="stretch" gap={0.5} mb={2}>
        {fired.map((g) => (
          <GuardRow key={g.name} guard={g} />
        ))}
      </VStack>

      {verdict.escalation_policy && verdict.escalation_policy.entries.length > 0 && (
        <Box mb={2} p={2} bg="orange.50" borderRadius="sm">
          <Text fontSize="xs" color="gray.700" mb={1}>
            Escalation policy v{verdict.escalation_policy.version} handed to Claude:
          </Text>
          {verdict.escalation_policy.entries.map((e) => (
            <Text key={e.guard} fontSize="xs" color="gray.700">
              <strong>{guardLabel(e.guard)}</strong> → default <strong>{e.final}</strong>: {e.instruction}
            </Text>
          ))}
        </Box>
      )}

      <Accordion.Root collapsible defaultValue={[]} mb={1}>
        <Accordion.Item value="guards" border="none">
          <Accordion.ItemTrigger px={0} py={1} _hover={{ bg: "transparent" }}>
            <Box flex="1" textAlign="left">
              <Text fontSize="xs" color="gray.600">
                {passed.length} guards passed
                {skipped.length > 0 ? `, ${skipped.length} not applicable` : ""}
              </Text>
            </Box>
            <Accordion.ItemIndicator />
          </Accordion.ItemTrigger>
          <Accordion.ItemContent>
            <Accordion.ItemBody px={0} pb={1}>
              <VStack align="stretch" gap={0.5}>
                {passed.map((g) => (
                  <GuardRow key={g.name} guard={g} />
                ))}
                {skipped.length > 0 && (
                  <Text fontSize="xs" color="gray.500">
                    Not applicable (no evidence):{" "}
                    {skipped.map((g) => guardLabel(g.name)).join(", ")}
                  </Text>
                )}
              </VStack>
            </Accordion.ItemBody>
          </Accordion.ItemContent>
        </Accordion.Item>
        <Accordion.Item value="judgments" border="none">
          <Accordion.ItemTrigger px={0} py={1} _hover={{ bg: "transparent" }}>
            <Box flex="1" textAlign="left">
              <Text fontSize="xs" color="gray.600">
                Jev judged {judgments.length} evidence items (yes ≥ {verdict.thresholds?.yes}, no ≤{" "}
                {verdict.thresholds?.no}, between is unsure)
              </Text>
            </Box>
            <Accordion.ItemIndicator />
          </Accordion.ItemTrigger>
          <Accordion.ItemContent>
            <Accordion.ItemBody px={0} pb={1}>
              <VStack align="stretch" gap={1}>
                {judgments.map((j) => (
                  <HStack key={j.id} gap={2} fontSize="xs" align="start">
                    <Badge
                      colorPalette={BAND_COLORS[j.band] ?? "gray"}
                      variant={j.band === "no" ? "subtle" : "solid"}
                      fontSize="xs"
                      minW="72px"
                      justifyContent="center"
                    >
                      {j.band} {(j.p * 100).toFixed(0)}%
                    </Badge>
                    <Text color="gray.700" flex={1}>
                      {FAMILY_LABELS[j.family] ?? j.family} · <strong>{j.ref}</strong>
                    </Text>
                  </HStack>
                ))}
              </VStack>
            </Accordion.ItemBody>
          </Accordion.ItemContent>
        </Accordion.Item>
        <Accordion.Item value="facts" border="none">
          <Accordion.ItemTrigger px={0} py={1} _hover={{ bg: "transparent" }}>
            <Box flex="1" textAlign="left">
              <Text fontSize="xs" color="gray.600">
                {facts.length} facts computed by code (no model)
              </Text>
            </Box>
            <Accordion.ItemIndicator />
          </Accordion.ItemTrigger>
          <Accordion.ItemContent>
            <Accordion.ItemBody px={0} pb={1}>
              <VStack align="stretch" gap={0.5}>
                {facts.map((f) => (
                  <HStack key={f.name} gap={2} fontSize="xs" align="start">
                    <Badge colorPalette="blue" variant="subtle" fontSize="xs" minW="72px" justifyContent="center">
                      {f.value === null ? "unknown" : String(f.value)}
                    </Badge>
                    <Text color="gray.700" flex={1}>
                      <strong>{f.label || f.name}</strong> · {f.detail}
                    </Text>
                  </HStack>
                ))}
              </VStack>
            </Accordion.ItemBody>
          </Accordion.ItemContent>
        </Accordion.Item>
      </Accordion.Root>
    </Box>
  );
}

function FastAnswerCard({
  fastAnswer,
  additionalToolsUsed,
  finalRecommendation,
  isStreaming,
}: {
  fastAnswer: FastAnswer;
  additionalToolsUsed?: string[];
  finalRecommendation?: FinalRecommendation | null;
  isStreaming?: boolean;
}) {
  const verdict = fastAnswer.verdict;
  if (!verdict) return null;
  const routed = verdict.routed_decision ?? verdict.decision;
  const wasRerouted = routed !== verdict.decision;
  const escalated = routed === "needs_llm_decision";
  const decomposed = verdict.mode === "decomposed";

  return (
    <Box
      mb={3}
      p={3}
      bg="white"
      borderRadius="md"
      border="1px solid"
      borderColor={`${VERDICT_COLORS[routed] ?? "gray"}.300`}
    >
      <HStack mb={2} flexWrap="wrap" gap={2}>
        <Badge colorPalette="gray" fontSize="xs">
          Fast answer · {decomposed ? "code guards + Jev judgments" : verdict.model}
        </Badge>
        <Badge colorPalette={VERDICT_COLORS[routed] ?? "gray"} fontSize="md" px={2}>
          {escalated ? "escalated to Claude" : decomposed ? verdict.outcome : routed}
        </Badge>
        <Text fontSize="xs" color="gray.600">
          {decomposed
            ? `${verdict.question_count ?? 0} Jev judgments in ${verdict.latency_ms} ms`
            : `confidence ${((verdict.confidence ?? 0) * 100).toFixed(0)}% · Jev ${verdict.latency_ms} ms`}{" "}
          · total {fastAnswer.total_ms} ms
        </Text>
      </HStack>

      {fastAnswer.customer && (
        <Text fontSize="xs" color="gray.700" mb={2}>
          {fastAnswer.request?.type
            ? String(fastAnswer.request.type).replace(/_/g, " ")
            : "request"}{" "}
          for <strong>{fastAnswer.customer.name}</strong>
          {fastAnswer.request?.requested_amount_usd
            ? ` · $${Number(fastAnswer.request.requested_amount_usd).toLocaleString()}`
            : ""}
          {fastAnswer.request?.category ? ` · ${fastAnswer.request.category}` : ""}
        </Text>
      )}

      {decomposed ? (
        <DecomposedTrace verdict={verdict} />
      ) : (
        <>
        {wasRerouted && (
          <Text fontSize="xs" color="orange.700" mb={2}>
            Jev&apos;s top pick was &quot;{verdict.decision}&quot;, but{" "}
            {verdict.routing?.reason}.
          </Text>
        )}

        <VStack align="stretch" gap={1} mb={2}>
          {Object.entries(verdict.probabilities ?? {}).map(([option, p]) => (
            <HStack key={option} gap={2}>
              <Text fontSize="xs" w="150px" color="gray.700">
                {option.replace(/_/g, " ")}
              </Text>
              <Box flex={1} h="8px" bg="gray.100" borderRadius="full">
                <Box
                  h="8px"
                  w={`${Math.round(p * 100)}%`}
                  bg={`${VERDICT_COLORS[option] ?? "gray"}.400`}
                  borderRadius="full"
                />
              </Box>
              <Text fontSize="xs" w="40px" textAlign="right" color="gray.700">
                {(p * 100).toFixed(0)}%
              </Text>
            </HStack>
          ))}
        </VStack>

        <Flex gap={1} flexWrap="wrap" mb={2}>
          {Object.entries(verdict.signals ?? {}).map(([name, p]) =>
            p === null ? (
              <Badge
                key={name}
                colorPalette="gray"
                variant="outline"
                fontSize="xs"
                title="Not asked: the evidence this signal depends on was empty"
              >
                {name.replace(/_/g, " ")} · no evidence
              </Badge>
            ) : (
              <Badge
                key={name}
                colorPalette={p >= 0.5 ? "purple" : "gray"}
                variant={p >= 0.5 ? "solid" : "subtle"}
                fontSize="xs"
              >
                {name.replace(/_/g, " ")} {(p * 100).toFixed(0)}%
              </Badge>
            ),
          )}
        </Flex>

        {verdict.item_judgments &&
          (verdict.item_judgments.precedents.length > 0 ||
            verdict.item_judgments.policies.length > 0) && (
            <Accordion.Root collapsible defaultValue={[]} mb={1}>
              <Accordion.Item value="items" border="none">
                <Accordion.ItemTrigger px={0} py={1} _hover={{ bg: "transparent" }}>
                  <Box flex="1" textAlign="left">
                    <Text fontSize="xs" color="gray.600">
                      How Jev judged each precedent (
                      {verdict.item_judgments.precedents.length}) and policy (
                      {verdict.item_judgments.policies.length})
                    </Text>
                  </Box>
                  <Accordion.ItemIndicator />
                </Accordion.ItemTrigger>
                <Accordion.ItemContent>
                  <Accordion.ItemBody px={0} pb={1}>
                    <VStack align="stretch" gap={1}>
                      {verdict.item_judgments.precedents.map((p) => (
                        <HStack key={p.id} gap={2} fontSize="xs" align="start">
                          <Badge
                            colorPalette={p.supports_approval >= 0.5 ? "green" : "red"}
                            fontSize="xs"
                            minW="72px"
                            justifyContent="center"
                          >
                            {p.supports_approval >= 0.5 ? "for" : "against"}{" "}
                            {(p.supports_approval * 100).toFixed(0)}%
                          </Badge>
                          <Text color="gray.700" flex={1}>
                            <strong>{p.decision_type}</strong> · {p.category} · {p.summary}
                          </Text>
                        </HStack>
                      ))}
                      {verdict.item_judgments.policies.map((p) => (
                        <HStack key={p.id} gap={2} fontSize="xs" align="start">
                          <Badge
                            colorPalette={p.violated >= 0.5 ? "red" : "gray"}
                            variant={p.violated >= 0.5 ? "solid" : "subtle"}
                            fontSize="xs"
                            minW="72px"
                            justifyContent="center"
                          >
                            violated {(p.violated * 100).toFixed(0)}%
                          </Badge>
                          <Text color="gray.700" flex={1}>
                            <strong>{p.name}</strong> · {p.description}
                          </Text>
                        </HStack>
                      ))}
                    </VStack>
                  </Accordion.ItemBody>
                </Accordion.ItemContent>
              </Accordion.Item>
            </Accordion.Root>
          )}
        </>
      )}

      <Accordion.Root collapsible defaultValue={[]}>
        <Accordion.Item value="evidence" border="none">
          <Accordion.ItemTrigger px={0} py={1} _hover={{ bg: "transparent" }}>
            <Box flex="1" textAlign="left">
              <Text fontSize="xs" color="gray.600">
                Evidence: {fastAnswer.retrievals?.length ?? 0} Cypher retrievals
                run up front (no tool-use LLM)
              </Text>
            </Box>
            <Accordion.ItemIndicator />
          </Accordion.ItemTrigger>
          <Accordion.ItemContent>
            <Accordion.ItemBody px={0} pb={1}>
              <VStack align="stretch" gap={0}>
                {fastAnswer.retrievals?.map((r) => (
                  <HStack key={r.name} gap={2} fontSize="xs">
                    <Badge
                      colorPalette={r.status === "ok" ? "blue" : "red"}
                      fontSize="xs"
                    >
                      {r.name}
                    </Badge>
                    <Text color="gray.600" flex={1}>
                      {r.description}
                    </Text>
                    <Text color="gray.500">
                      {r.count} rows · {r.ms} ms
                    </Text>
                  </HStack>
                ))}
              </VStack>
            </Accordion.ItemBody>
          </Accordion.ItemContent>
        </Accordion.Item>
      </Accordion.Root>

      <Box mt={2} pt={2} borderTop="1px dashed" borderColor="gray.200">
        {!isStreaming && finalRecommendation && (
          <Box mb={2}>
            <FinalRecommendationRow recommendation={finalRecommendation} />
          </Box>
        )}
        {isStreaming ? (
          <HStack fontSize="xs" color="gray.600">
            <Spinner size="xs" />
            <Text>
              {escalated
                ? "Escalated. Claude is making the final call from the same evidence…"
                : "Claude is explaining the verdict from the same evidence…"}
            </Text>
          </HStack>
        ) : additionalToolsUsed && additionalToolsUsed.length > 0 ? (
          <HStack fontSize="xs" color="gray.700" flexWrap="wrap">
            <Text>Claude reached for additional tools:</Text>
            {additionalToolsUsed.map((t) => (
              <Badge key={t} colorPalette="blue" fontSize="xs">
                {t}
              </Badge>
            ))}
          </HStack>
        ) : (
          <Text fontSize="xs" color="gray.600">
            {escalated
              ? "Claude decided from the pre-collected evidence alone. No additional tools used."
              : "Claude explained the verdict from the pre-collected evidence alone. No additional tools used."}
          </Text>
        )}
      </Box>
    </Box>
  );
}

// Agent context disclosure component
function AgentContextDisclosure({
  agentContext,
}: {
  agentContext: AgentContext;
}) {
  return (
    <Box mb={3} pb={3} borderBottom="1px solid" borderColor="gray.200">
      <Accordion.Root collapsible defaultValue={[]}>
        <Accordion.Item value="agent-context" border="none">
          <Accordion.ItemTrigger px={0} py={2} _hover={{ bg: "transparent" }}>
            <Box flex="1" textAlign="left">
              <HStack>
                <Badge colorPalette="teal" fontSize="xs">
                  Agent Context
                </Badge>
                <Text fontSize="xs" color="gray.600">
                  View system prompt and configuration
                </Text>
              </HStack>
            </Box>
            <Accordion.ItemIndicator />
          </Accordion.ItemTrigger>
          <Accordion.ItemContent>
            <Accordion.ItemBody px={0} pb={2}>
              <VStack align="stretch" gap={3}>
                <Box
                  p={3}
                  bg="teal.50"
                  borderRadius="md"
                  border="1px solid"
                  borderColor="teal.200"
                >
                  <VStack align="stretch" gap={3}>
                    {/* Model */}
                    <Box>
                      <Text
                        fontSize="xs"
                        color="gray.600"
                        fontWeight="semibold"
                        mb={1}
                      >
                        Model:
                      </Text>
                      <Badge colorPalette="teal" fontSize="xs">
                        {agentContext.model}
                      </Badge>
                    </Box>

                    {/* MCP Server */}
                    <Box>
                      <Text
                        fontSize="xs"
                        color="gray.600"
                        fontWeight="semibold"
                        mb={1}
                      >
                        MCP Server:
                      </Text>
                      <Badge colorPalette="purple" fontSize="xs">
                        {agentContext.mcp_server}
                      </Badge>
                    </Box>

                    {/* Available Tools */}
                    <Box>
                      <Text
                        fontSize="xs"
                        color="gray.600"
                        fontWeight="semibold"
                        mb={1}
                      >
                        Available Tools:
                      </Text>
                      <Flex gap={1} flexWrap="wrap">
                        {agentContext.available_tools.map((tool, idx) => (
                          <Badge key={idx} colorPalette="blue" fontSize="xs">
                            {tool}
                          </Badge>
                        ))}
                      </Flex>
                    </Box>

                    {/* System Prompt */}
                    <Box>
                      <Text
                        fontSize="xs"
                        color="gray.600"
                        fontWeight="semibold"
                        mb={1}
                      >
                        System Prompt:
                      </Text>
                      <Code
                        display="block"
                        whiteSpace="pre-wrap"
                        p={2}
                        borderRadius="md"
                        fontSize="xs"
                        bg="white"
                        maxH="300px"
                        overflowY="auto"
                      >
                        {agentContext.system_prompt}
                      </Code>
                    </Box>
                  </VStack>
                </Box>
              </VStack>
            </Accordion.ItemBody>
          </Accordion.ItemContent>
        </Accordion.Item>
      </Accordion.Root>
    </Box>
  );
}

// Tool calls disclosure component
function ToolCallsDisclosure({ toolCalls }: { toolCalls: ToolCall[] }) {
  if (!toolCalls || toolCalls.length === 0) return null;

  return (
    <Box mb={3} pb={3} borderBottom="1px solid" borderColor="gray.200">
      <Accordion.Root collapsible defaultValue={[]}>
        <Accordion.Item value="tool-calls" border="none">
          <Accordion.ItemTrigger px={0} py={2} _hover={{ bg: "transparent" }}>
            <Box flex="1" textAlign="left">
              <HStack>
                <Badge colorPalette="purple" fontSize="xs">
                  Tool Calls ({toolCalls.length})
                </Badge>
                <Text fontSize="xs" color="gray.600">
                  View tool parameters and results
                </Text>
              </HStack>
            </Box>
            <Accordion.ItemIndicator />
          </Accordion.ItemTrigger>
          <Accordion.ItemContent>
            <Accordion.ItemBody px={0} pb={2}>
              <VStack align="stretch" gap={3}>
                {toolCalls.map((toolCall, index) => (
                  <Box
                    key={index}
                    p={3}
                    bg="purple.50"
                    borderRadius="md"
                    border="1px solid"
                    borderColor="purple.200"
                  >
                    <ToolCallDisplay toolCall={toolCall} index={index} />
                  </Box>
                ))}
              </VStack>
            </Accordion.ItemBody>
          </Accordion.ItemContent>
        </Accordion.Item>
      </Accordion.Root>
    </Box>
  );
}

// Chat message bubble component
function ChatMessageBubble({
  message,
  isStreaming,
  onDecisionClick,
  onNodeClick,
}: {
  message: MessageWithGraph;
  isStreaming?: boolean;
  onDecisionClick: (decision: Decision) => void;
  onNodeClick: (nodeId: string, labels: string[]) => void;
}) {
  const isUser = message.role === "user";

  return (
    <Box
      alignSelf={isUser ? "flex-end" : "flex-start"}
      maxW="85%"
      w={
        message.graphData ||
        (!isUser &&
          (message.agentContext ||
            message.toolCalls?.length ||
            message.fastAnswer?.applicable))
          ? "100%"
          : "auto"
      }
      minW={
        !isUser && (message.agentContext || message.toolCalls?.length)
          ? "60%"
          : "auto"
      }
    >
      <Box
        bg={isUser ? "brand.500" : "bg.subtle"}
        color={isUser ? "white" : "inherit"}
        px={4}
        py={3}
        borderRadius="lg"
        borderBottomRightRadius={isUser ? "sm" : "lg"}
        borderBottomLeftRadius={isUser ? "lg" : "sm"}
      >
        {/* Fast typed verdict from Jev (for assistant messages) */}
        {!isUser && message.fastAnswer?.applicable && (
          <FastAnswerCard
            fastAnswer={message.fastAnswer}
            additionalToolsUsed={message.additionalToolsUsed}
            finalRecommendation={message.finalRecommendation}
            isStreaming={isStreaming}
          />
        )}

        {/* Agent context disclosure (for assistant messages) */}
        {!isUser && message.agentContext && (
          <AgentContextDisclosure agentContext={message.agentContext} />
        )}

        {/* Tool calls disclosure (for assistant messages) */}
        {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
          <ToolCallsDisclosure toolCalls={message.toolCalls} />
        )}

        {/* Message content */}
        <Flex align="flex-start" gap={2}>
          {isUser ? (
            <Text whiteSpace="pre-wrap" fontSize="sm" flex={1}>
              {message.content}
            </Text>
          ) : (
            <Box flex={1} fontSize="sm" className="markdown-content">
              <ReactMarkdown
                components={{
                  p: ({ children }) => (
                    <Text mb={2} _last={{ mb: 0 }}>
                      {children}
                    </Text>
                  ),
                  strong: ({ children }) => (
                    <Text as="strong" fontWeight="bold">
                      {children}
                    </Text>
                  ),
                  em: ({ children }) => (
                    <Text as="em" fontStyle="italic">
                      {children}
                    </Text>
                  ),
                  ul: ({ children }) => (
                    <Box as="ul" pl={4} mb={2}>
                      {children}
                    </Box>
                  ),
                  ol: ({ children }) => (
                    <Box as="ol" pl={4} mb={2}>
                      {children}
                    </Box>
                  ),
                  li: ({ children }) => (
                    <Box as="li" mb={1}>
                      {children}
                    </Box>
                  ),
                  code: ({ children, className }) => {
                    const isInline = !className;
                    return isInline ? (
                      <Code fontSize="xs" px={1}>
                        {children}
                      </Code>
                    ) : (
                      <Box
                        as="pre"
                        bg="bg.emphasized"
                        p={3}
                        borderRadius="md"
                        overflow="auto"
                        mb={2}
                        fontSize="xs"
                      >
                        <code>{children}</code>
                      </Box>
                    );
                  },
                  h1: ({ children }) => (
                    <Text fontSize="lg" fontWeight="bold" mb={2} mt={3}>
                      {children}
                    </Text>
                  ),
                  h2: ({ children }) => (
                    <Text fontSize="md" fontWeight="bold" mb={2} mt={3}>
                      {children}
                    </Text>
                  ),
                  h3: ({ children }) => (
                    <Text fontSize="sm" fontWeight="bold" mb={1} mt={2}>
                      {children}
                    </Text>
                  ),
                  blockquote: ({ children }) => (
                    <Box
                      borderLeftWidth={3}
                      borderLeftColor="gray.300"
                      pl={3}
                      my={2}
                      color="gray.600"
                    >
                      {children}
                    </Box>
                  ),
                }}
              >
                {stripFinalRecommendation(message.content)}
              </ReactMarkdown>
            </Box>
          )}
          {isStreaming && <Spinner size="xs" color="brand.500" />}
        </Flex>
      </Box>

      {/* Inline graph visualization */}
      {message.graphData && message.graphData.nodes.length > 0 && (
        <Box mt={2}>
          <InlineGraph
            graphData={message.graphData}
            height="250px"
            onNodeClick={onNodeClick}
          />
          <Text fontSize="xs" color="gray.500" mt={1}>
            {message.graphData.nodes.length} nodes,{" "}
            {message.graphData.relationships.length} relationships
          </Text>
        </Box>
      )}
    </Box>
  );
}

// Suggestion chip component
function SuggestionChip({
  text,
  onClick,
}: {
  text: string;
  onClick: () => void;
}) {
  return (
    <Text
      fontSize="sm"
      color="brand.600"
      cursor="pointer"
      _hover={{ textDecoration: "underline" }}
      onClick={onClick}
    >
      → {text}
    </Text>
  );
}

// Send icon component
function SendIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}
