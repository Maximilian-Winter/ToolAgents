import { useState, useRef, useEffect, type FormEvent } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useRooms, useMessages, useSendMessage, useCreateRoom } from '../hooks/useChat';
import { useSSE } from '../hooks/useSSE';
import { Avatar, Button, EmptyState, Select } from '../components/ui';
import { MESSAGE_TYPES, TYPE_COLORS, senderColor, formatTime, canGroup } from '../lib/chat';
import { cx } from '../lib/cx';
import type { Message } from '../api/types';
import MentionRenderer from '../components/MentionRenderer';
import styles from './ChatRoom.module.css';

/* ─── Message Bubble (Discord-style: flat, grouped) ─── */
function MessageBubble({
  msg,
  showHeader,
}: {
  msg: Message;
  showHeader: boolean;
}) {
  const color = senderColor(msg.sender);
  const typeColor = TYPE_COLORS[msg.message_type] || TYPE_COLORS.statement;
  const isNonStatement = msg.message_type !== 'statement';

  return (
    <div
      className={cx(
        showHeader ? styles.messageGroup : styles.msgContinuation,
        isNonStatement && styles.typeIndicator
      )}
      style={isNonStatement ? { borderLeftColor: typeColor } : undefined}
    >
      {showHeader && (
        <div className={styles.messageGroupHeader}>
          <Avatar name={msg.sender} size="sm" />
          <span className={styles.msgSender} style={{ color }}>
            {msg.sender}
          </span>
          <span className={styles.msgTime}>{formatTime(msg.created_at)}</span>
        </div>
      )}
      <div className={styles.msgContent}>
        {isNonStatement && (
          <span
            className={styles.typeBadge}
            style={{
              color: typeColor,
              background: `color-mix(in srgb, ${typeColor} 15%, transparent)`,
            }}
          >
            {msg.message_type}
          </span>
        )}
        {msg.reply_to && (
          <span className={styles.replyTag}>replying to #{msg.reply_to}</span>
        )}
        {msg.to && <span className={styles.mentionTag}>@{msg.to}</span>}
        <MentionRenderer text={msg.content} />
        {msg.reactions && msg.reactions.length > 0 && (
          <div className={styles.reactions}>
            {msg.reactions.map((r) => (
              <span key={r.emoji} className={styles.reaction}>
                {r.emoji} {r.count}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── Main ChatRoom Component ─── */
export default function ChatRoom() {
  const { slug, room } = useParams<{ slug: string; room: string }>();
  const { data: rooms } = useRooms(slug);
  const { data: messages, isLoading: messagesLoading } = useMessages(slug, room);
  const sendMessage = useSendMessage(slug, room);
  const createRoom = useCreateRoom(slug);

  const { typing } = useSSE(slug, room);

  const [content, setContent] = useState('');
  const [sender, setSender] = useState('user');
  const [messageType, setMessageType] = useState('statement');
  const [showNewRoom, setShowNewRoom] = useState(false);
  const [newRoomName, setNewRoomName] = useState('');

  const listEndRef = useRef<HTMLDivElement>(null);
  const typingNames = Object.keys(typing).filter((n) => n !== sender);

  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = (e: FormEvent) => {
    e.preventDefault();
    if (!content.trim()) return;
    sendMessage.mutate(
      { sender, content: content.trim(), message_type: messageType },
      { onSuccess: () => setContent('') }
    );
  };

  const handleCreateRoom = (e: FormEvent) => {
    e.preventDefault();
    if (!newRoomName.trim()) return;
    createRoom.mutate(
      { name: newRoomName.trim() },
      {
        onSuccess: () => {
          setNewRoomName('');
          setShowNewRoom(false);
        },
      }
    );
  };

  // Collect unique senders for the member sidebar
  const members = messages
    ? [...new Set(messages.map((m) => m.sender))]
    : [];

  return (
    <div className={styles.container}>
      {/* ── Room Sidebar ── */}
      <div className={styles.roomSidebar}>
        <div className={styles.roomHeader}>
          <span>Rooms</span>
          <button
            className={styles.addRoomBtn}
            onClick={() => setShowNewRoom(!showNewRoom)}
            title="New room"
          >
            +
          </button>
        </div>

        {showNewRoom && (
          <form onSubmit={handleCreateRoom}>
            <input
              className={styles.newRoomInput}
              value={newRoomName}
              onChange={(e) => setNewRoomName(e.target.value)}
              placeholder="room-name"
              autoFocus
              onBlur={() => {
                if (!newRoomName) setShowNewRoom(false);
              }}
            />
          </form>
        )}

        {rooms?.map((r) => (
          <Link
            key={r.id}
            to={`/projects/${slug}/chat/${r.name}`}
            className={cx(
              styles.roomItem,
              r.name === room && styles.roomActive
            )}
          >
            <span className={styles.roomHash}>#</span>
            {r.name}
          </Link>
        ))}
      </div>

      {/* ── Chat Area ── */}
      <div className={styles.chatArea}>
        {room ? (
          <>
            <div className={styles.chatHeader}>
              <span className={styles.chatHeaderHash}>#</span>
              {room}
            </div>

            <div className={styles.messageList}>
              {messagesLoading && (
                <EmptyState message="Loading messages..." />
              )}
              {messages?.map((msg, i) => {
                const prev = i > 0 ? messages[i - 1] : null;
                const showHeader = !prev || !canGroup(prev, msg);
                return (
                  <MessageBubble key={msg.id} msg={msg} showHeader={showHeader} />
                );
              })}
              {messages && messages.length === 0 && (
                <EmptyState
                  icon="💬"
                  message="No messages yet. Start the conversation."
                />
              )}
              <div ref={listEndRef} />
              {typingNames.length > 0 && (
                <div className={styles.typing}>
                  {typingNames.join(', ')}{' '}
                  {typingNames.length === 1 ? 'is' : 'are'} typing...
                </div>
              )}
            </div>

            {/* ── Message Input ── */}
            <form className={styles.inputArea} onSubmit={handleSend}>
              <div className={styles.inputControls}>
                <input
                  className={styles.newRoomInput}
                  style={{ width: 90, margin: 0 }}
                  value={sender}
                  onChange={(e) => setSender(e.target.value)}
                  placeholder="Sender"
                  title="Sender name"
                />
                <Select
                  selectSize="sm"
                  value={messageType}
                  onChange={(e) => setMessageType(e.target.value)}
                  options={MESSAGE_TYPES.map((t) => ({ value: t, label: t }))}
                  style={{ width: 110 }}
                />
              </div>
              <div className={styles.inputRow}>
                <textarea
                  className={styles.messageTextarea}
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder={`Message #${room}...`}
                  rows={1}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend(e);
                    }
                  }}
                />
                <Button
                  type="submit"
                  size="sm"
                  disabled={sendMessage.isPending}
                >
                  Send
                </Button>
              </div>
            </form>
          </>
        ) : (
          <div className={styles.emptyChat}>
            <EmptyState
              icon="💬"
              message="Select a room to start chatting"
            />
          </div>
        )}
      </div>

      {/* ── Member Sidebar (shown when room is active) ── */}
      {room && members.length > 0 && (
        <div className={styles.memberSidebar}>
          <div className={styles.memberHeader}>
            Members — {members.length}
          </div>
          {members.map((name) => (
            <div key={name} className={styles.memberItem}>
              <Avatar name={name} size="sm" />
              <span className={styles.memberName}>{name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
