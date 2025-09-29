import StatIcon from "./StatIcon";

const CARD_WIDTH = 90;
const CARD_HEIGHT = 120;

export default function Card({ img, cost, attack, health, x, y }) {
    return (
        <g transform={`translate(${x}, ${y})`}>
            <image href={img} x="0" y="0" width={CARD_WIDTH} height={CARD_HEIGHT} />

            {/* フレーム画像（カード本体の上に重ねる） */}
            <image href="frame.png" x="0" y="0" width={CARD_WIDTH} height={CARD_HEIGHT} />

            {cost !== null && (
                <StatIcon icon="cost_icon.png" value={cost} x={0} y={0} />
            )}
            {attack !== null && (
                <StatIcon icon="attack_icon.png" value={attack} x={0} y={CARD_HEIGHT - 40} />
            )}
            {health !== null && (
                <StatIcon icon="health_icon.png" value={health} x={CARD_WIDTH - 40} y={CARD_HEIGHT - 40} />
            )}
        </g>
    );
}
