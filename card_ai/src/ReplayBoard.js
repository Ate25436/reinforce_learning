import { useEffect, useState } from "react";
import Card from "./Card";

export default function ReplayBoard() {
    const [battleLog, setBattleLog] = useState([]);
    const [cardData, setCardData] = useState({});
    const [index, setIndex] = useState(0);

    useEffect(() => {
        Promise.all([
            fetch(process.env.PUBLIC_URL + "/battle_log.json").then(res => res.json()),
            fetch(process.env.PUBLIC_URL + "/card_data.json").then(res => res.json())
        ]).then(([log, cards]) => {
            setBattleLog(log);
            setCardData(cards);
        });
    }, []);

    if (battleLog.length === 0) return <div>Loading...</div>;

    const board = battleLog[index];
    const cardWidth = 120;
    const spacing = 20;

    const arrangeRow = (ids, startX, y) =>
        ids.map((id, i) => {
            const card = cardData[id];
            console.log(id);
            return {
                id,
                ...card,
                x: startX + i * (cardWidth + spacing),
                y
            };
        });

    const arrangeField = (slots, startX, y) =>
        slots
            .map((slot, i) => {
                const [atk, hp, id] = slot;
                if (atk === 0 && hp === 0 ) return null;
                const baseCard = cardData[id];
                return {
                    id: `field_${i}_${id}`,
                    ...baseCard,
                    attack: atk,
                    health: hp,
                    x: startX + i * (cardWidth + spacing),
                    y
                };
            })
            .filter(Boolean);

    const playerHand = arrangeRow(board.agent_0.hand, 50, 450);
    const playerField = arrangeField(board.agent_0.field, 150, 300);
    const enemyField = arrangeField(board.agent_1.field, 150, 150);
    const enemyHand = arrangeRow(board.agent_1.hand, 50, 20);

    return (
        <div>
            <svg width="1200" height="800">
                <image
                    href="board.png"
                    x="0"
                    y="0"
                    width="1000"
                    height="600"
                />
                <text x="50" y="420" fill="green" fontSize="20" fontWeight="bold">
                    Life: {board.agent_0.health}
                </text>
                <text x="50" y="440" fill="blue" fontSize="20" fontWeight="bold">
                    PP: {board.agent_0.PP}
                </text>
                <text x="50" y="170" fill="green" fontSize="20" fontWeight="bold">
                    Life: {board.agent_1.health}
                </text>
                <text x="50" y="190" fill="blue" fontSize="20" fontWeight="bold">
                    PP: {board.agent_1.PP}
                </text>

                {playerHand.map((c, i) => (
                    <Card key={`playerHand_${c.id}_${i}`} {...c} />
                ))}
                {playerField.map((c, i) => (
                    <Card key={`playerField_${c.id}_${i}`} {...c} />
                ))}
                {enemyField.map((c, i) => (
                    <Card key={`enemyField_${c.id}_${i}`} {...c} />
                ))}
                {enemyHand.map((c, i) => (
                    <Card key={`enemyHand_${c.id}_${i}`} {...c} />
                ))}
            </svg>
            <div style={{ marginTop: "10px" }}>
                <button onClick={() => setIndex(Math.max(0, index - 1))}>◀ 前へ</button>
                <span style={{ margin: "0 20px" }}>
                    {index + 1} / {battleLog.length}
                </span>
                <button onClick={() => setIndex(Math.min(battleLog.length - 1, index + 1))}>
                    次へ ▶
                </button>
            </div>
        </div>
    );
}
